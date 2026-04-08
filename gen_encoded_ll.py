import torch
import torch.backends.cudnn as cudnn
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import cv2
from torchvision.transforms import ToTensor
import numpy as np
import random

from bayes_opt.gppllie_helper import GPPLLIEHelper
from bayes_opt.utils import fiFindByWildcard

from distill.distill_model import BODistillation

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
    out_dir = os.path.join(inp_dir, 'encoded_ll')
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
    to_tensor = ToTensor()

    bo_distill = BODistillation().to(device).eval()

    # --- 3. Main Processing Loop ---
    i = 0
    for _key in common_keys:
        save_path = os.path.join(out_dir, _key + ".pt")
        if i == -1: exit()

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
            gppllie.get_invariant_features(img_tensor)
            y = gppllie.y.squeeze(0).detach().to("cpu", dtype=torch.float32).contiguous()
            print(f"=== gppllie.y: {gppllie.y.shape}")

            torch.save(
                {"y": y, "lr_path": lr_path, "H": img_rgb.shape[0], "W": img_rgb.shape[1]},
                save_path
            )
            print(f"=== encoded ll saved to {save_path}")

        except Exception as e:
            print(f"!!!!! Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            # runs whether success or failure
            for name in ["img_tensor"]:
                if name in locals():
                    del locals()[name]
            torch.cuda.empty_cache()
        i += 1


if __name__ == "__main__":
    input_dirs = [
        'dataset/DICM',
        'dataset/LIME', 
        'dataset/MEF', 
        'dataset/NPE',
        'dataset/LOLv1/test', 
        'dataset/LOLv2_syn/test'
    ]

    for input_dir in input_dirs:
        main(input_dir)
