import torch
import torch.backends.cudnn as cudnn
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import cv2
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
import uuid
import pandas as pd
import numpy as np
import random
# --- NEW IMPORTS FOR BO ---
from ax.service.ax_client import AxClient
import pyiqa  # Library for Image Quality Assessment

from bayes_opt.bayes_opt_helper import BayesOptimizationHelper
from bayes_opt.gppllie_helper import GPPLLIEHelper
from bayes_opt.utils import fiFindByWildcard, plot_bo_trials

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


def main(inp_dir):
    # --- 1. Directory Setup ---
    lr_dir = os.path.join(inp_dir, 'low')
    global_prior_dir = os.path.join(inp_dir, 'global_score')
    local_prior_dir = os.path.join(inp_dir, 'local_prior')
    out_dir = os.path.join(inp_dir, 'outputs_bo_v2_weights_12_compare_env2')
    os.makedirs(out_dir, exist_ok=True)

    lr_paths = []
    for ext in ('*.jpg', '*.JPG', '*.bmp', '*.png'):
        lr_paths += fiFindByWildcard(os.path.join(lr_dir, ext))

    global_prior_paths = fiFindByWildcard(os.path.join(global_prior_dir, '*.pt'))
    local_prior_paths = fiFindByWildcard(os.path.join(local_prior_dir, '*.pt'))

    def _key_from_path(p):
        # use filename without extension; keep spaces; normalize case for matching only
        return os.path.splitext(os.path.basename(p))[0].lower()

    lr_map = {_key_from_path(p): p for p in lr_paths}
    g_map = {_key_from_path(p): p for p in global_prior_paths}
    l_map = {_key_from_path(p): p for p in local_prior_paths}
    common_keys = sorted(set(lr_map) & set(g_map) & set(l_map))

    # just in case, report missing
    missing_g = sorted(set(lr_map) - set(g_map))
    missing_l = sorted(set(lr_map) - set(l_map))
    assert not missing_g and not missing_l and len(missing_g) == len(missing_l) == 0, (
        f"[ERROR] Missing global priors for: {missing_g[:10]}{'...' if len(missing_g) > 10 else ''};"
        f"[ERROR] Missing local  priors for: {missing_l[:10]}{'...' if len(missing_l) > 10 else ''}"
    )
    if not common_keys:
        print(f"=== Matched triplets: {len(common_keys)} / LL images: {len(lr_paths)}")
        print(f"=== common_keys: {common_keys}")

    # --- 2. Load All Models ---
    device = torch.device('cuda:0')
    print("=== Initializing Models...")
    # gppllie = GPPLLIEHelper(model_path='weight_lolv1.pth', device=device)
    gppllie = GPPLLIEHelper(model_path='weight_lolv2_syn.pth', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    to_tensor = ToTensor()

    # --- 3. Main Processing Loop ---
    i = 0
    for _key in common_keys:
        if i == 3: exit()

        lr_path = lr_map[_key]
        global_path = g_map[_key]
        local_path = l_map[_key]
        print(f"=== lr_path: {lr_path}")
        print(f"=== global_path: {global_path}")
        print(f"=== local_path: {local_path}")

        filename = os.path.basename(lr_path)
        print(f"=== Processing: {filename}")

        try:
            img_rgb = cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)
            img_tensor = to_tensor(img_rgb).unsqueeze(0).to(device)
            # raw metrics (low light imgs)
            with torch.no_grad():
                raw_niqe = niqe_metric(img_tensor)
                raw_musiq = musiq_metric(img_tensor)
            print(
                f"---> Raw Metrics: "
                f"---> raw niqe: {raw_niqe}; raw musiq: {raw_musiq}"
            )

            global_prior_base = torch.load(global_path, map_location=device)
            local_prior_base = torch.load(local_path, map_location=device)
            gppllie.get_invariant_features(img_tensor)
            sr_raw = gppllie.get_denoised_and_decoded_img(
                global_prior=global_prior_base,
                local_prior=local_prior_base
            )
            sr_raw_clamped = torch.clamp(sr_raw, 0, 1)
            save_img_path = os.path.join(out_dir, f"WO_BO_{filename}")
            save_image(sr_raw_clamped, save_img_path)
            print(f"=== Saved to {save_img_path}")

            # ==========================================
            #      BAYESIAN OPTIMIZATION
            # ==========================================
            ax_client = AxClient(verbose_logging=False)
            bo = BayesOptimizationHelper(
                device, global_prior_base, local_prior_base,
                gppllie, niqe_metric, musiq_metric, ax_client
            )
            searching_params = [
                {"name": "global_scale", "type": "range", "bounds": [0.8, 1.2], "value_type": "float"},
                {"name": "local_scale", "type": "range", "bounds": [0.8, 1.2], "value_type": "float"},
                {"name": "local_gamma", "type": "range", "bounds": [0.9, 1.1], "value_type": "float"},
            ]
            uid = uuid.uuid4().hex[:8]
            exp_name = f"multi_bo_{os.path.splitext(filename)[0]}_{uid}"
            baseline_params = {
                "global_scale": 1.0, "local_scale": 1.0, "local_gamma": 1.0,
            }

            sr_clamped = bo.generation_sr(baseline_params)
            wo_bo_niqe, wo_bo_musiq = bo.evaluate(sr_clamped)
            print(
                f"---> Metrics after GPP-LLIE w/o BO: "
                f"---> niqe: {wo_bo_niqe}; musiq: {wo_bo_musiq}"
            )
            bo.create_experiment_with_constraints(
                searching_params=searching_params,
                name=exp_name,
                baseline_musiq_val=wo_bo_musiq
            )

            baseline_trial_index = bo.add_baseline_trial(
                wo_bo_niqe, wo_bo_musiq, baseline_params
            )
            bo.run_n_trials(n_trials=20)
            best_row, df = bo.get_best_params()
            final_niqe, final_musiq = best_row["niqe"], best_row["musiq"]
            print(
                f"---> Metrics after GPP-LLIE with BO: "
                f"---> niqe: {final_niqe}; musiq: {final_musiq}"
            )

            # Generate Final "With BO" Image
            sr_final_clamped = bo.generation_sr(best_row)
            bo_img_path = os.path.join(out_dir, f"BO_{filename}")
            save_image(sr_final_clamped, bo_img_path)
            print(f"=== Saved to {bo_img_path}")

            # Save Logs & Visualize
            df.to_csv(os.path.join(out_dir, f"{exp_name}.csv"), index=False)
            plot_path = os.path.join(out_dir, f"{exp_name}_pareto.png")
            plot_bo_trials(df, plot_path, best_row['trial_index'], baseline_trial_index)
            print(f"=== Plot saved to {plot_path}")

            summary = {
                "experiment_name": exp_name,
                "raw_niqe": float(raw_niqe.item()),
                "wo_bo_niqe": float(wo_bo_niqe),
                "final_niqe": float(final_niqe),
                "final_musiq": float(final_musiq),
                "best_trial_index": int(best_row["trial_index"]),
                "best_global_scale": float(best_row["global_scale"]),
                "best_local_scale": float(best_row["local_scale"]),
                "best_local_gamma": float(best_row["local_gamma"])
            }
            pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"{exp_name}_summary.csv"), index=False)

            comparison = {
                "filename": filename,
                "raw_niqe": float(raw_niqe.item()),
                "without_bo_niqe": float(wo_bo_niqe),
                "with_bo_niqe": float(final_niqe),
                "niqe_improvement": float(wo_bo_niqe - final_niqe),
                "raw_musiq": float(raw_musiq.item()),
                "without_bo_musiq": float(wo_bo_musiq),
                "with_bo_musiq": float(final_musiq),
                "musiq_improvement": float(final_musiq - wo_bo_musiq)
            }
            pd.DataFrame([comparison]).to_csv(os.path.join(out_dir, f"{filename}_comparison.csv"), index=False)

        except Exception as e:
            print(f"!!!!! Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # runs whether success or failure
            for name in ["img_tensor", "global_prior_base", "local_prior_base",
                         "sr_raw", "sr_raw_clamped", "sr_final_clamped",
                         "ax_client", "bo", "df", "best_row"]:
                if name in locals():
                    del locals()[name]
            torch.cuda.empty_cache()

        i += 1

    print("=== All images processed successfully! ===")


if __name__ == "__main__":
    input_dir = 'dataset/DICM'
    main(input_dir)
