import os
import sys
import json
import time
import torch
import importlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, DiffusionPipeline
from ptp import invutils, frq_ptputils

'''
"CompVis/stable-diffusion-v1-4"
"runwayml/stable-diffusion-v1-5"
'''


# setup the device and backbone model
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
NUM_INFER_STEPS = 50
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                           beta_schedule="scaled_linear", clip_sample=False, 
                           set_alpha_to_one=False, steps_offset=1)
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler).to(DEVICE)
model.scheduler.set_timesteps(NUM_INFER_STEPS)

g_seed = 8888
g_cpu = torch.Generator().manual_seed(8888)



# select an image and load for editing
img_names = ["image_89.jpeg"]
img_dir = "./DATASET"
i_name = img_names[0]
img_path = os.path.join(img_dir, i_name)
img_np = invutils.load_512(img_path)
img_latn = invutils.img2latn(img_np, model, DEVICE)
save_dir = './outputs/test/'

# the vae-reconstruct image is saved to show the difference caused by vae encoder
os.makedirs(save_dir, exist_ok=True)
img_latn_rec = invutils.latn2img(img_latn, model)[0]
Image.fromarray(img_latn_rec).save(os.path.join(save_dir, i_name[:-4]+"vae-rec.png"))
uncond_emb = invutils.encode_text("", model)
xT, xts = invutils.ddim_inversion_null_fixpt(img_latn, model, uncond_emb, save_all=True, FP_STEPS=5, 
                                                  INV_STEPS=NUM_INFER_STEPS,)



'''
    The target prompt should follow the principle mentioned in the paper.
    For cases of changing a single object, use a simple one that only describes the target(effect).
    For cases of changing a object while the editing effect affects backgrounds, use a target prompt that contains the description of surroundings.
'''
prompt = ["a plane"]


res_latents = frq_ptputils.frq_img_gen(model, prompt, g_seed=g_seed,
                     latent=xT, mod_guidance_frq =True,
                    save_dir=save_dir, guidance_scale=7.5, 
                    TS=(921, 861, 681, 481), FS=(4, 6, 8, 8, 16), HS=(32, 32, 32, 32, 32, 32),
                    filter_shape = 'sq', 
                    # remove = True,
                    # generate_mask=True, gen_MSK_THRS=0.025,
                    # guide_mask=True, 
                    # reverse_mask=True,
                    num_infer_steps = NUM_INFER_STEPS,
                    clear_low = True,
                    record_time = True
                    )



