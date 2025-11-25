import torch
import torch.backends.cudnn as cudnn
import glob
import os
import natsort
import cv2
from torchvision.utils import save_image
from torchvision.transforms import ToTensor

# --- Existing Project Imports ---
from model_incontext_revise import DiT_incontext_revise
from diffusion import create_diffusion
from vae.autoencoder import AutoencoderKL
from vae.cond_encoder import CondEncoder
from vae.encoder_decoder import Decoder2

# --- NEW IMPORTS FOR BO ---
from ax.service.ax_client import AxClient, ObjectiveProperties
import pyiqa  # Library for Image Quality Assessment

# Optimized settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

EPS = 1e-6


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def main(inp_dir):
    # --- 1. Directory Setup ---
    lr_dir = os.path.join(inp_dir, 'low')
    global_prior_dir = os.path.join(inp_dir, 'global_score')
    local_prior_dir = os.path.join(inp_dir, 'local_hist_prior')
    out_dir = os.path.join(inp_dir, 'outputs_bo')
    os.makedirs(out_dir, exist_ok=True)

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    global_prior_paths = fiFindByWildcard(os.path.join(global_prior_dir, '*.pt'))
    local_prior_paths = fiFindByWildcard(os.path.join(local_prior_dir, '*.pt'))

    device = torch.device('cuda:0')

    # --- 2. Load All Models ---
    print("Loading models...")
    state_dict = torch.load('weight_lolv1.pth', map_location=device)

    # Diffusion Backbone
    model = DiT_incontext_revise().to(device)
    model.load_state_dict(state_dict['dit'], strict=True)
    model.eval()

    # VAE
    vae = AutoencoderKL().to(device)
    vae.load_state_dict(state_dict['vae'], strict=True)
    vae.eval()

    # Conditional Encoder
    cond_lq = CondEncoder().to(device)
    cond_lq.load_state_dict(state_dict['cond'], strict=True)
    cond_lq.eval()

    # Refinement Decoder
    second_decoder = Decoder2().to(device)
    second_decoder.load_state_dict(state_dict['second_decoder'], strict=True)
    second_decoder.eval()

    # --- 3. Load Evaluation Metric for BO ---
    print("Loading IQ Metric (NIQE)...")
    # IMPORTANT: For NIQE, Lower is Better.
    # For MUSIQ, Higher is Better.
    metric_scorer = pyiqa.create_metric('niqe', device=device)

    # Diffusion Scheduler
    diffusion_val = create_diffusion(str(25))
    to_tensor = ToTensor()

    # --- 4. Main Processing Loop ---
    for lr_path, global_path, local_path in zip(lr_paths, global_prior_paths, local_prior_paths):
        filename = os.path.basename(lr_path)
        print(f"Processing: {filename}")

        # Pre-load Data
        img_bgr = cv2.imread(lr_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
        print(f"=== img_tensor: {img_tensor.shape}")

        global_prior_base = torch.load(global_path, map_location=device)
        local_prior_base = torch.load(local_path, map_location=device)

        # Both the BO loop and the Final Save will use this same 'norm_local_prior_base'
        min_val = local_prior_base.min()
        max_val = local_prior_base.max()
        if (max_val - min_val) > EPS:
            norm_local_prior_base = (local_prior_base - min_val) / (max_val - min_val)
        else:
            norm_local_prior_base = local_prior_base

        print(f"Check Prior Range: Min={min_val.item():.4f}, Max={max_val.item():.4f}")

        b, c, h, w = img_tensor.shape

        # Pre-calculate invariant features
        with torch.no_grad():
            y, enc_feat = cond_lq(img_tensor, True)
            print(f"=== y: {y.shape}")
            print(f"=== enc_feat: {enc_feat.shape}")

            # Fix the random noise 'z'
            latent_h, latent_w = h // 4, w // 4
            z_fixed = torch.randn(1, 3, latent_h, latent_w, device=device)

        # ==========================================
        #      BAYESIAN OPTIMIZATION SETUP
        # ==========================================

        def generation_objective(parameterization):
            """
            The function that BO calls repeatedly.
            """
            g_scale = parameterization.get("global_scale", 1.0)
            l_scale = parameterization.get("local_scale", 1.0)
            l_gamma = parameterization.get("local_gamma", 1.0)

            # 1. Modify the Prior
            adjusted_global_prior = global_prior_base * g_scale

            # Use the PRE-NORMALIZED map here (Safety + Consistency)
            adjusted_local_prior = ((norm_local_prior_base + EPS) ** l_gamma) * l_scale

            with torch.no_grad():
                model_kwargs = dict(
                    y=y,
                    vis=adjusted_global_prior,
                    q_map=adjusted_local_prior
                )

                # 2. Run Diffusion
                samples = diffusion_val.p_sample_loop(
                    model.forward, z_fixed.shape, z_fixed, clip_denoised=False,
                    model_kwargs=model_kwargs, progress=False, device=device
                )

                # 3. Decode & Refine
                dec_feat = vae.decode(samples, mid_feat=True)
                sr_result = second_decoder(samples, dec_feat, enc_feat)

                # 4. Calculate Score
                sr_clamped = torch.clamp(sr_result, 0, 1)
                score = metric_scorer(sr_clamped).item()

            return score

        # Initialize Ax Client
        ax_client = AxClient(verbose_logging=False)

        # Define Search Space
        ax_client.create_experiment(
            name=f"optimize_{filename}",
            parameters=[
                {
                    "name": "global_scale",
                    "type": "range",
                    "bounds": [0.5, 1.5],
                    "value_type": "float",
                },
                {
                    "name": "local_scale",
                    "type": "range",
                    "bounds": [0.5, 2.0],
                    "value_type": "float",
                },
                {
                    "name": "local_gamma",
                    "type": "range",
                    "bounds": [0.5, 3.0],
                    "value_type": "float",
                }
            ],
            # minimize=True for NIQE
            # minimize=False for MUSIQ
            objectives={"quality_score": ObjectiveProperties(minimize=True)},
        )

        # Run Optimization Loop
        num_trials = 15
        print(f"  > Running BO for {num_trials} trials...")

        for i in range(num_trials):
            params, trial_index = ax_client.get_next_trial()
            try:
                score = generation_objective(params)
                ax_client.complete_trial(trial_index=trial_index, raw_data=score)
                print(f"    Trial {i}: G={params['global_scale']:.2f}, "
                      f"L={params['local_scale']:.2f}, Gamma={params['local_gamma']:.2f}"
                      f" -> NIQE={score:.4f}")
            except Exception as e:
                print(f"    Trial {i} Failed: {e}")
                ax_client.log_trial_failure(trial_index=trial_index)

        # ==========================================
        #      FINAL GENERATION WITH BEST PARAM
        # ==========================================
        best_params, _ = ax_client.get_best_parameters()
        print(f"  > Best: G={best_params['global_scale']:.2f}, "
              f"L={best_params['local_scale']:.2f}, "
              f"Gamma={best_params['local_gamma']:.2f}")

        with torch.no_grad():
            adjusted_global_prior = global_prior_base * best_params['global_scale']

            # Retrieve params
            final_l_scale = best_params['local_scale']
            final_l_gamma = best_params['local_gamma']

            # Apply EXACTLY the same logic as in objective_function
            # We use the pre-calculated norm_local_prior_base
            adjusted_local_prior = ((norm_local_prior_base + EPS) ** final_l_gamma) * final_l_scale

            model_kwargs = dict(y=y, vis=adjusted_global_prior, q_map=adjusted_local_prior)
            samples = diffusion_val.p_sample_loop(
                model.forward, z_fixed.shape, z_fixed, clip_denoised=False,
                model_kwargs=model_kwargs, progress=True, device=device
            )
            dec_feat = vae.decode(samples, mid_feat=True)
            sr_final = second_decoder(samples, dec_feat, enc_feat)

        save_img_path = os.path.join(out_dir, filename)
        save_image(sr_final, save_img_path)
        print(f"  > Saved to {save_img_path}")


if __name__ == "__main__":
    input_dir = 'dataset/LOLv2_syn/Test'
    main(input_dir)