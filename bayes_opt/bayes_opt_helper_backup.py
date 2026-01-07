import torch
import torch.backends.cudnn as cudnn
from ax.service.ax_client import ObjectiveProperties

from bayes_opt.constants import EPS, NIQE_MIN, NIQE_MAX, MUSIQ_MIN, MUSIQ_MAX
from bayes_opt.utils import _to_float

# Optimized settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class BayesOptimizationHelper:
    def __init__(self, device, global_prior, local_prior,
                 gppllie,
                 niqe_metric, musiq_metric, ax_client,
                 ref_img_tensor):
        self.device = device
        self.ax_client = ax_client
        self.ref_img_tensor = ref_img_tensor

        # Reuse shared scorers (no need to recreate per image)
        self.niqe_metric = niqe_metric
        self.musiq_metric = musiq_metric

        self.global_prior = global_prior
        self.local_prior = local_prior
        self.normalized_local_prior = self._normalize_local_prior(local_prior)

        # gppllie
        self.gpplie = gppllie
        self.encoded_img = gppllie.y
        self.encoded_features = gppllie.enc_feat
        self.diffusion_framework = gppllie.diffusion_val
        self.dit = gppllie.model
        self.rand_noise = gppllie.z_fixed
        self.vae = gppllie.vae
        self.second_decoder = gppllie.second_decoder

    @staticmethod
    def calculate_contrast_rms(image_tensor):
        """Calculate RMS contrast - standard deviation of pixel intensities"""
        # Handle batch dimension
        if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
            image_tensor = image_tensor.squeeze(0)

        # Convert to grayscale for contrast calculation
        if image_tensor.shape[0] == 3:  # RGB image
            gray = 0.2989 * image_tensor[0] + 0.5870 * image_tensor[1] + 0.1140 * image_tensor[2]
        else:
            gray = image_tensor[0] if image_tensor.dim() == 3 else image_tensor

        return _to_float(torch.std(gray))

    @staticmethod
    def _normalize_local_prior(local_prior):
        # Both the BO loop and the Final Save will use this same 'norm_local_prior_base'
        min_val = local_prior.min()
        max_val = local_prior.max()
        if (max_val - min_val) > EPS:
            return (local_prior - min_val) / (max_val - min_val)
        return local_prior

    @staticmethod
    def get_lum(img):
        """
        BT.601
        """
        return 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]

    def highlight_clip_ratio(self, img, thr=0.98):
        """
        Fraction of pixels whose luminance is near-white.
        Larger => more over-exposure / 'night -> day' risk.
        img: (1,3,H,W) in [0,1]
        """
        lum = self.get_lum(img)
        return _to_float((lum > thr).float().mean())

    def lum_ratio(self, enhanced_img):
        """Mean luminance ratio relative to ref image."""
        ref = self.get_lum(self.ref_img_tensor).mean()
        new = self.get_lum(enhanced_img).mean()
        return _to_float(new / (ref + EPS))

    def apply_priors(self, params):
        # params is dict-like (Ax params dict or pandas row)
        g = float(params["global_scale"])
        lg = float(params["local_gamma"])
        ls = float(params["local_scale"])

        adjusted_global = self.global_prior * g
        adjusted_local = ((self.normalized_local_prior + EPS) ** lg) * ls

        return adjusted_global, adjusted_local

    def generation_objective(self, params):
        """
        The function that BO calls repeatedly.
        Notes:
          - We keep z_fixed fixed (from GPPLLIEHelper.get_invariant_features) for fair comparisons.
          - We do NOT re-seed every call (can bias optimization to one deterministic noise realization).
        """
        # 1. Modify the Prior
        adjusted_global_prior, adjusted_local_prior = self.apply_priors(params)

        with torch.no_grad():
            sr_result = self.gpplie.get_denoised_and_decoded_img(
                global_prior=adjusted_global_prior,
                local_prior=adjusted_local_prior
            )

            # Clamp
            sr_clamped = torch.clamp(sr_result, 0, 1)

            # --- METRICS ---
            niqe_val = _to_float(self.niqe_metric(sr_clamped))  # lower is better
            musiq_val = _to_float(self.musiq_metric(sr_clamped))  # higher is better
            clip_ratio = self.highlight_clip_ratio(sr_clamped, thr=0.98)
            contrast_score = _to_float(self.calculate_contrast_rms(sr_clamped))
            lum_ratio = self.lum_ratio(sr_clamped)

        return niqe_val, musiq_val, clip_ratio, contrast_score, lum_ratio

    def create_experiment_with_constraints(self, name, searching_params,
                                           baseline_musiq_val,
                                           baseline_clip_ratio,
                                           baseline_contrast=None,
                                           baseline_lum_ratio=None):
        """
        Create experiment with constraints (night-preserving):
          - MUSIQ not worse than baseline
          - clip_ratio close to baseline (avoid over-exposure)
          - lum_ratio not too large (avoid night->day)
          - optionally constrain contrast inflation
        """
        # Adaptive clip margin:
        # - if baseline already has noticeable clipping, tighten; else allow a bit more room
        clip_margin = 0.005 if baseline_clip_ratio > 0.01 else 0.01
        clip_limit = baseline_clip_ratio + clip_margin

        # Adaptive luminance ratio upper bound
        # (if baseline_lum_ratio is around 1.0 (ref), allow +35%; if ref is extremely dark, still cap)
        if baseline_lum_ratio is None:
            lum_upper = 1.35
        else:
            lum_upper = max(1.25, min(1.45, baseline_lum_ratio + 0.35))

        # Optional contrast upper bound: allow +20%
        contrast_upper_limit = (baseline_contrast * 1.2) if baseline_contrast is not None else None

        outcome_constraints = [
            f"musiq >= {baseline_musiq_val - 3.0}",
            f"clip_ratio <= {clip_limit}",
            f"lum_ratio <= {lum_upper}",
        ]
        if contrast_upper_limit is not None:
            outcome_constraints.append(f"contrast <= {contrast_upper_limit}")

        self.ax_client.create_experiment(
            name=name,
            parameters=searching_params,
            objectives={
                "niqe": ObjectiveProperties(minimize=True),
            },
            outcome_constraints=outcome_constraints,
            overwrite_existing_experiment=True
        )

    def add_baseline_trial(self, baseline_params):
        baseline_niqe, baseline_musiq, clip_ratio, baseline_contrast, baseline_lum_ratio = \
            self.generation_objective(baseline_params)

        print(
            f"[Baseline] G={baseline_params['global_scale']:.2f}, "
            f"L={baseline_params['local_scale']:.2f}, "
            f"Gamma={baseline_params['local_gamma']:.2f} "
            f"-> NIQE={baseline_niqe:.4f}, MUSIQ={baseline_musiq:.4f}, "
            f"ClipRatio={clip_ratio:.6f}, LumRatio={baseline_lum_ratio:.4f}, "
            f"Contrast={baseline_contrast:.6f}"
        )

        # use baseline as the 1st known trial
        _, baseline_trial_index = self.ax_client.attach_trial(baseline_params)
        raw_data = {
            "niqe": (baseline_niqe, 0.0),
            "musiq": (baseline_musiq, 0.0),
            "clip_ratio": (clip_ratio, 0.0),
            "lum_ratio": (baseline_lum_ratio, 0.0),
            "contrast": (baseline_contrast, 0.0),
        }
        self.ax_client.complete_trial(trial_index=baseline_trial_index, raw_data=raw_data)

        return baseline_trial_index, clip_ratio, baseline_contrast, baseline_lum_ratio

    def run_n_trials(self, n_trials):
        print(f"=== Running MO-BO for {n_trials} trials with baseline...")

        for i in range(n_trials):
            params, trial_index = self.ax_client.get_next_trial()
            try:
                niqe_val, musiq_val, clip_ratio, contrast_score, lum_ratio = \
                    self.generation_objective(params)

                raw_data = {
                    "niqe": (niqe_val, 0.0),
                    "musiq": (musiq_val, 0.0),
                    "clip_ratio": (clip_ratio, 0.0),
                    "lum_ratio": (lum_ratio, 0.0),
                    "contrast": (contrast_score, 0.0),
                }
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

                print(
                    f"    Trial {i + 1}: "
                    f"G={params['global_scale']:.3f}, "
                    f"L={params['local_scale']:.3f}, "
                    f"Gamma={params['local_gamma']:.3f} "
                    f"-> NIQE={niqe_val:.4f}, MUSIQ={musiq_val:.4f}, "
                    f"ClipRatio={clip_ratio:.6f}, LumRatio={lum_ratio:.4f}, "
                    f"Contrast={contrast_score:.6f}"
                )
            except Exception as e:
                print(f"    Trial {i + 1} Failed: {e}")
                self.ax_client.log_trial_failure(trial_index=trial_index)

    def get_best_params(self, baseline_trial_index):
        """Select best params considering both metrics AND night-preservation."""
        print(f'=== All Trials DF:\n{self.ax_client.get_trials_data_frame()}')
        df = self.ax_client.get_trials_data_frame()
        df = df[df["trial_status"] == "COMPLETED"].copy()

        needed = ["niqe", "musiq", "clip_ratio", "lum_ratio", "contrast"]
        df = df.dropna(subset=needed)

        # Baseline stats
        base = df[df["trial_index"] == baseline_trial_index]
        baseline_clip = float(base["clip_ratio"].iloc[0]) if not base.empty else 0.0
        baseline_lum_ratio = float(base["lum_ratio"].iloc[0]) if not base.empty else 1.0

        # Normalize NIQE and MUSIQ for weighted score
        df["niqe_norm"] = (df["niqe"] - NIQE_MIN) / (NIQE_MAX - NIQE_MIN)
        df["musiq_norm"] = (MUSIQ_MAX - df["musiq"]) / (MUSIQ_MAX - MUSIQ_MIN)

        # Night penalties relative to baseline
        df["clip_penalty"] = (df["clip_ratio"] - baseline_clip).clip(lower=0.0) * 200.0
        df["lum_penalty"] = (df["lum_ratio"] - baseline_lum_ratio).clip(lower=0.0) * 5.0

        # You can optionally penalize too much contrast inflation
        df["contrast_penalty"] = (df["contrast"] - df["contrast"].median()).clip(lower=0.0) * 0.0

        # Weighted score (lower is better)
        w_niqe = 0.5
        w_musiq = 0.3
        w_night = 0.2

        df["score"] = (
                w_niqe * df["niqe_norm"] +
                w_musiq * df["musiq_norm"] +
                w_night * (df["clip_penalty"] + df["lum_penalty"] + df["contrast_penalty"])
        )

        best_row = df.loc[df["score"].idxmin()]

        print(
            f"[Best weighted params] trial={int(best_row['trial_index'])}, "
            f"G={best_row['global_scale']:.3f}, L={best_row['local_scale']:.3f}, "
            f"Gamma={best_row['local_gamma']:.3f} -> Score={best_row['score']:.6f}\n"
            f"    NIQE={best_row['niqe']:.4f}, MUSIQ={best_row['musiq']:.4f}, "
            f"ClipRatio={best_row['clip_ratio']:.6f}, LumRatio={best_row['lum_ratio']:.4f}"
        )

        return best_row, df
