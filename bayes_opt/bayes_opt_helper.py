import torch
import torch.backends.cudnn as cudnn
from ax.service.ax_client import ObjectiveProperties
import numpy as np
import random
from bayes_opt.constants import EPS, NIQE_MIN, NIQE_MAX, MUSIQ_MIN, MUSIQ_MAX
from bayes_opt.utils import _to_float

# Optimized settings
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class BayesOptimizationHelper:
    def __init__(self, device, global_prior, local_prior,
                 gppllie,
                 niqe_metric, musiq_metric, ax_client):
        self.device = device
        self.ax_client = ax_client

        # Reuse shared scorers (no need to recreate per image)
        self.niqe_metric = niqe_metric
        self.musiq_metric = musiq_metric

        self.global_prior = global_prior
        self.local_prior = local_prior
        self.normalized_local_prior = self._normalize_local_prior(local_prior)

        # gppllie
        self.gppllie = gppllie
        self.encoded_img = gppllie.y
        self.encoded_features = gppllie.enc_feat
        self.diffusion_framework = gppllie.diffusion_val
        self.dit = gppllie.model
        self.rand_noise = gppllie.z_fixed
        self.vae = gppllie.vae
        self.second_decoder = gppllie.second_decoder

    @staticmethod
    def _normalize_local_prior(local_prior):
        # Both the BO loop and the Final Save will use this same 'norm_local_prior_base'
        min_val = local_prior.min()
        max_val = local_prior.max()
        if (max_val - min_val) > EPS:
            return (local_prior - min_val) / (max_val - min_val)
        return local_prior

    def apply_priors(self, params):
        # params is dict-like (Ax params dict or pandas row)
        g = float(params["global_scale"])
        lg = float(params["local_gamma"])
        ls = float(params["local_scale"])

        adjusted_global = self.global_prior * g
        adjusted_local = ((self.normalized_local_prior + EPS) ** lg) * ls

        return adjusted_global, adjusted_local

    def generation_sr(self, params):
        # Modify the Prior
        adjusted_global_prior, adjusted_local_prior = self.apply_priors(params)
        with torch.no_grad():
            sr_result = self.gppllie.get_denoised_and_decoded_img(
                global_prior=adjusted_global_prior,
                local_prior=adjusted_local_prior
            )
            # Clamp
            sr_clamped = torch.clamp(sr_result, 0, 1)
        return sr_clamped

    def evaluate(self, sr_clamped):
        with torch.no_grad():
            niqe_val = _to_float(self.niqe_metric(sr_clamped))  # lower is better
            musiq_val = _to_float(self.musiq_metric(sr_clamped))  # higher is better
        return niqe_val, musiq_val

    def create_experiment_with_constraints(self, name, searching_params,
                                           baseline_musiq_val):
        outcome_constraints = [
            f"musiq >= {baseline_musiq_val - 3.0:.6f}"
        ]
        self.ax_client.create_experiment(
            name=name,
            parameters=searching_params,
            objectives={
                "niqe": ObjectiveProperties(minimize=True),
            },
            outcome_constraints=outcome_constraints,
            tracking_metric_names=["musiq"],
            overwrite_existing_experiment=True
        )

    def add_baseline_trial(self, baseline_niqe, baseline_musiq, baseline_params):
        print(
            f"[Baseline] G={baseline_params['global_scale']:.2f}, "
            f"L={baseline_params['local_scale']:.2f}, "
            f"Gamma={baseline_params['local_gamma']:.2f} "
        )

        # use baseline as the 1st known trial
        _, baseline_trial_index = self.ax_client.attach_trial(baseline_params)
        raw_data = {
            "niqe": (baseline_niqe, 0.0),
            "musiq": (baseline_musiq, 0.0)
        }
        self.ax_client.complete_trial(trial_index=baseline_trial_index, raw_data=raw_data)

        return baseline_trial_index

    def run_n_trials(self, n_trials):
        print(f"=== Running BO for {n_trials} trials with baseline...")

        for i in range(n_trials):
            params, trial_index = self.ax_client.get_next_trial()
            try:
                sr_clamped = self.generation_sr(params)
                niqe_val, musiq_val = self.evaluate(sr_clamped)

                raw_data = {
                    "niqe": (niqe_val, 0.0),
                    "musiq": (musiq_val, 0.0)
                }
                self.ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

                print(
                    f"    Trial {i + 1}: "
                    f"G={params['global_scale']:}, "
                    f"L={params['local_scale']}, "
                    f"Gamma={params['local_gamma']} "
                    f"-> NIQE={niqe_val}, MUSIQ={musiq_val}, "
                )
            except Exception as e:
                print(f"    Trial {i + 1} Failed: {e}, params={params}")
                self.ax_client.log_trial_failure(trial_index=trial_index)

    def get_best_params(self):
        """Select best params considering both metrics AND night-preservation."""
        print(f'=== All Trials DF:\n{self.ax_client.get_trials_data_frame()}')
        df = self.ax_client.get_trials_data_frame()
        df = df[df["trial_status"] == "COMPLETED"].copy()

        needed = ["niqe", "musiq"]
        df = df.dropna(subset=needed)

        # Normalize NIQE and MUSIQ for weighted score
        df["niqe_norm"] = (df["niqe"] - NIQE_MIN) / (NIQE_MAX - NIQE_MIN)
        df["musiq_norm"] = (MUSIQ_MAX - df["musiq"]) / (MUSIQ_MAX - MUSIQ_MIN)

        # Weighted score (lower is better)
        w_niqe, w_musiq = 0.7, 0.3
        df["score"] = (
                w_niqe * df["niqe_norm"] +
                w_musiq * df["musiq_norm"]
        )

        best_row = df.loc[df["score"].idxmin()]
        print(
            f"[Best weighted params] trial={int(best_row['trial_index'])}, "
            f"G={best_row['global_scale']}, L={best_row['local_scale']}, "
            f"Gamma={best_row['local_gamma']} -> Score={best_row['score']}\n"
            f"    NIQE={best_row['niqe']}, MUSIQ={best_row['musiq']}"
        )

        return best_row, df
