import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import glob
import os
from model_incontext_revise import DiT_incontext_revise
from diffusion import create_diffusion
from vae.autoencoder import AutoencoderKL
from vae.cond_encoder import CondEncoder
from vae.encoder_decoder import Decoder2
from torchvision.utils import save_image
import natsort
from torchvision.transforms import ToTensor
import cv2
import numpy as np
from download import load_model


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def t(array):
    return torch.Tensor(
        np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)
    ) / 255


def rgb(t):
    return (np.clip((
                        t[0] if len(t.shape) == 4 else t
                    ).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255
            ).astype(np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def main(inp_dir):
    lr_dir = os.path.join(inp_dir, 'low')
    os.makedirs(lr_dir, exist_ok=True)
    global_prior_dir = os.path.join(inp_dir, 'global_score')
    os.makedirs(global_prior_dir, exist_ok=True)
    local_prior_dir = os.path.join(inp_dir, 'local_hist_prior')
    os.makedirs(local_prior_dir, exist_ok=True)
    out_dir = os.path.join(inp_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    global_prior_paths = fiFindByWildcard(os.path.join(global_prior_dir, '*.pt'))
    local_prior_paths = fiFindByWildcard(os.path.join(local_prior_dir, '*.pt'))

    device = torch.device('cuda:0')
    state_dict = torch.load('weight_lolv1.pth')

    # Transformer based on diffusions
    # diffusion Transformer backbone, with GPP-LN and LPP-Attn inside
    model = DiT_incontext_revise()
    model.load_state_dict(state_dict['dit'], strict=True)
    # ckpt_dit = './experiments/GPP_LLIE_LOLv1_dit/models/1/1.pth'  # self trained
    # state_dict = load_model(ckpt_dit)
    # model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    # Variational Auto-encoder
    # KL: KL divergence
    vae = AutoencoderKL()
    vae.load_state_dict(state_dict['vae'], strict=True)
    vae = vae.to(device)

    # Conditional Encoder
    # encodes the input + priors into conditioning tokens
    cond_lq = CondEncoder()
    cond_lq.load_state_dict(state_dict['cond'], strict=True)
    # ckpt_condencoder = './experiments/GPP_LLIE_LOLv1_dit/models/1/1_condencoder.pth'  # self trained
    # state_dict = load_model(ckpt_condencoder)
    # cond_lq.load_state_dict(state_dict, strict=True)
    cond_lq = cond_lq.to(device)

    # Decoder2
    # second-stage refinement decoder
    second_decoder = Decoder2()
    second_decoder.load_state_dict(state_dict['second_decoder'], strict=True)
    # ckpt_second_decoder = './experiments/GPP_LLIE_LOLv1_decoder2/models/1/1_seconddecoder.pth'  # self trained
    # state_dict = load_model(ckpt_second_decoder)
    # second_decoder.load_state_dict(state_dict, strict=True)
    second_decoder = second_decoder.to(device)

    model.eval()
    diffusion_val = create_diffusion(str(25))  # number of sample steps

    to_tensor = ToTensor()

    for lr_path, global_path, local_path, test_index in zip(lr_paths,
                                                            global_prior_paths,
                                                            local_prior_paths,
                                                            range(len(lr_paths))):
        # y = t(imread(lr_path)).to(device)
        img = to_tensor(cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)).unsqueeze(0)
        # print(y.shape)
        global_prior = torch.load(global_path, map_location="cuda:0").to(device)
        local_prior = torch.load(local_path, map_location="cuda:0").to(device)
        print(f"=== local_prior: {local_prior.shape}")

        b, c, h, w = img.shape
        # use less memo and run faster without calculating gradients
        with torch.no_grad():
            y, enc_feat = cond_lq(img.to(device), True)
            latent_size_h = h // 4
            latent_size_w = w // 4
            z = torch.randn(1, 3, latent_size_h, latent_size_w, device=device)
            print(f"=== z: {z.shape}")
            model_kwargs = dict(y=y, vis=global_prior, q_map=local_prior)

            # Sample images:
            # reverse diffusion process
            samples = diffusion_val.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )

            dec_feat = vae.decode(samples, mid_feat=True)

            sr = second_decoder(samples, dec_feat, enc_feat)

        save_img_path = os.path.join(out_dir, os.path.basename(lr_path))
        save_image(sr, save_img_path)


if __name__ == "__main__":
    # update the input dir, which at least contains
    # such sub-folder: low, global_score, local_prior
    input_dir = 'dataset/LOLv2_syn/Test'

    main(input_dir)
