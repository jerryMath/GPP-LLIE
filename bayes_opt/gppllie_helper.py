import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
from model_incontext_revise import DiT_incontext_revise
from diffusion import create_diffusion
from vae.autoencoder import AutoencoderKL
from vae.cond_encoder import CondEncoder
from vae.encoder_decoder import Decoder2

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

class GPPLLIEHelper:
    def __init__(self, model_path, device):
        self._state_dict = torch.load(model_path, map_location=device)
        self._device = device

        self.model = None
        self.vae = None
        self.cond_lq = None
        self.second_decoder = None
        self.diffusion_val = None
        self._extract()

        self.y = None
        self.enc_feat = None
        self.z_fixed = None

        self._invariant_ready = False

    def _extract(self):
        print("=== Loading models...")
        # Diffusion Backbone
        self.model = DiT_incontext_revise().to(self._device).eval()
        self.model.load_state_dict(self._state_dict['dit'], strict=True)
        # VAE
        self.vae = AutoencoderKL().to(self._device).eval()
        self.vae.load_state_dict(self._state_dict['vae'], strict=True)
        # Conditional Encoder
        self.cond_lq = CondEncoder().to(self._device).eval()
        self.cond_lq.load_state_dict(self._state_dict['cond'], strict=True)
        # Refinement Decoder
        self.second_decoder = Decoder2().to(self._device).eval()
        self.second_decoder.load_state_dict(self._state_dict['second_decoder'], strict=True)
        # Diffusion Scheduler
        self.diffusion_val = create_diffusion(str(25))

    def _check_invariant_ready(self):
        if not self._invariant_ready:
            raise RuntimeError(
                "Invariant features not initialized. "
                "Call get_invariant_features(img) before sampling."
            )

    def get_invariant_features(self, img_tensor, seed=0):
        print("=== Getting invariant features...")
        b, c, h, w = img_tensor.shape
        # Fix the random noise 'z' deterministically for fair comparisons
        g = torch.Generator(device=self._device)
        g.manual_seed(seed)
        with torch.no_grad():
            self.y, self.enc_feat = self.cond_lq(img_tensor, True)
            latent_h, latent_w = h // 4, w // 4
            self.z_fixed = torch.randn(1, 3, latent_h, latent_w, device=self._device, generator=g)

        self._invariant_ready = True

    def get_denoised_and_decoded_img(self, global_prior, local_prior):
        print("=== Getting denoised and decoded image via DDIM...")
        self._check_invariant_ready()
        with torch.no_grad():
            model_kwargs = dict(y=self.y, vis=global_prior, q_map=local_prior)
            samples = self.diffusion_val.ddim_sample_loop(
                self.model.forward,
                self.z_fixed.shape,
                self.z_fixed,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=self._device,
                eta=0.0,  # 0.0 = deterministic DDIM
            )
            dec_feat = self.vae.decode(samples, mid_feat=True)
            sr_final = self.second_decoder(samples, dec_feat, self.enc_feat)
        return sr_final
