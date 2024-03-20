import sys
sys.path.append('versatile_diffusion')
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-diff_str", "--diff_str",help="Diffusion Strength",default=0.75)
parser.add_argument("-mix_str", "--mix_str",help="Mixing Strength",default=0.4)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
strength = float(args.diff_str)
mixing = float(args.mix_str)
gpu_offset = 2


def regularize_image(x):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x

cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)    


# Might require editing the GPU assignments due to Memory issues
net.clip.cuda(0 + gpu_offset)
net.autokl.cuda(0 + gpu_offset)

#net.model.cuda(1)
sampler = sampler(net)
#sampler.model.model.cuda(1)
#sampler.model.cuda(1)
batch_size = 1

# pred_text = np.load('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral.npy'.format(sub))
# pred_text = np.load('data/predicted_features/subj{:02d}/nsd_cliptext_predtest_nsdgeneral_assumehrf.npy'.format(sub))
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext.npy')
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_200ms.npy')
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_400ms.npy')
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_600ms.npy')
pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/sub10/thingseeg2_regress_cliptext_800ms.npy')
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_200ms_1regressor.npy')
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_400ms_1regressor.npy')
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_600ms_1regressor.npy')
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_800ms_1regressor.npy')
# pred_text = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_cliptext_null.npy')
# pred_text = np.load('cache/thingseeg2/predicted_embeddings/thingseeg2_regress_cliptext_avg1_200ms.npy')
# pred_text = np.load('cache/thingseeg2/predicted_embeddings/thingseeg2_regress_cliptext_avg1_400ms.npy')
# pred_text = np.load('cache/thingseeg2/predicted_embeddings/thingseeg2_regress_cliptext_avg1_600ms.npy')
# pred_text = np.load('cache/thingseeg2/predicted_embeddings/thingseeg2_regress_cliptext_avg1_800ms.npy')
pred_text = torch.tensor(pred_text).half().cuda(1 + gpu_offset)

# pred_vision = np.load('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral.npy'.format(sub))
# pred_vision = np.load('data/predicted_features/subj{:02d}/nsd_clipvision_predtest_nsdgeneral_assumehrf.npy'.format(sub))
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_200ms.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_400ms.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_600ms.npy')
pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/sub10/thingseeg2_regress_clipvision_800ms.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_200ms_1regressor.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_400ms_1regressor.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_600ms_1regressor.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_800ms_1regressor.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_regress_clipvision_null.npy')
# pred_vision = np.load('cache/thingseeg2_preproc/predicted_embeddings/thingseeg2_dnn_clipvision_800ms.npy')
# pred_vision = np.load('cache/thingseeg2/predicted_embeddings/thingseeg2_regress_clipvision_avg1_200ms.npy')
# pred_vision = np.load('cache/thingseeg2/predicted_embeddings/thingseeg2_regress_clipvision_avg1_400ms.npy')
# pred_vision = np.load('cache/thingseeg2/predicted_embeddings/thingseeg2_regress_clipvision_avg1_600ms.npy')
# pred_vision = np.load('cache/thingseeg2/predicted_embeddings/thingseeg2_regress_clipvision_avg1_800ms.npy')
pred_vision = torch.tensor(pred_vision).half().cuda(1 + gpu_offset)


n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'
net.autokl.half()

torch.manual_seed(0)
for im_id in range(len(pred_vision)):

    # zim = Image.open('results/vdvae/subj{:02d}/{}.png'.format(sub,im_id))
    # zim = Image.open('results/thingseeg2_preproc/vdvae/{}.png'.format(im_id))
    # zim = Image.open('results/thingseeg2_preproc/vdvae_200ms/{}.png'.format(im_id))
    # zim = Image.open('results/thingseeg2_preproc/vdvae_400ms/{}.png'.format(im_id))
    # zim = Image.open('results/thingseeg2_preproc/vdvae_600ms/{}.png'.format(im_id))
    zim = Image.open('results/thingseeg2_preproc/sub10/vdvae_800ms/{}.png'.format(im_id))
    # zim = Image.open('results/thingseeg2_preproc/vdvae_null/{}.png'.format(im_id))
    # zim = Image.open('results/thingseeg2/vdvae_avg1_200ms/{}.png'.format(im_id))
    # zim = Image.open('results/thingseeg2/vdvae_avg1_400ms/{}.png'.format(im_id))
    # zim = Image.open('results/thingseeg2/vdvae_avg1_600ms/{}.png'.format(im_id))
    # zim = Image.open('results/thingseeg2/vdvae_avg1_800ms/{}.png'.format(im_id))
   
    zim = regularize_image(zim)
    zin = zim*2 - 1
    zin = zin.unsqueeze(0).cuda(0 + gpu_offset).half()

    init_latent = net.autokl_encode(zin)
    
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    #strength=0.75
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    device = 'cuda:' + str(gpu_offset)
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))
    #z_enc,_ = sampler.encode(init_latent.cuda(1).half(), c.cuda(1).half(), torch.tensor([t_enc]).to(sampler.model.model.diffusion_model.device))

    dummy = ''
    utx = net.clip_encode_text(dummy)
    utx = utx.cuda(1 + gpu_offset).half()
    
    dummy = torch.zeros((1,3,224,224)).cuda(0 + gpu_offset)
    uim = net.clip_encode_vision(dummy)
    uim = uim.cuda(1 + gpu_offset).half()
    
    z_enc = z_enc.cuda(1 + gpu_offset)

    h, w = 512,512
    shape = [n_samples, 4, h//8, w//8]

    cim = pred_vision[im_id].unsqueeze(0)
    ctx = pred_text[im_id].unsqueeze(0)
    
    #c[:,0] = u[:,0]
    #z_enc = z_enc.cuda(1).half()
    
    sampler.model.model.diffusion_model.device='cuda:' + str(1 + gpu_offset)
    sampler.model.model.diffusion_model.half().cuda(1 + gpu_offset)
    #mixing = 0.4
    
    z = sampler.decode_dc(
        x_latent=z_enc,
        first_conditioning=[uim, cim],
        second_conditioning=[utx, ctx],
        t_start=t_enc,
        unconditional_guidance_scale=scale,
        xtype='image', 
        first_ctype='vision',
        second_ctype='prompt',
        mixed_ratio=(1-mixing), )
    
    z = z.cuda(0 + gpu_offset).half()
    x = net.autokl_decode(z)
    color_adj='None'
    #color_adj_to = cin[0]
    color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
    color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
    color_adj_keep_ratio = 0.5
    
    if color_adj_flag and (ctype=='vision'):
        x_adj = []
        for xi in x:
            color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
            xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
            x_adj.append(xi_adj)
        x = x_adj
    else:
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]
    

    # x[0].save('results/versatile_diffusion/subj{:02d}/{}.png'.format(sub,im_id))
    # x[0].save('results/versatile_diffusion/subj{:02d}_assumehrf/{}.png'.format(sub,im_id))
        
    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion/{}.png'.format(im_id))
        
    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_200ms/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_200ms/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_200ms/{}.png'.format(im_id))

    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_400ms/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_400ms/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_400ms/{}.png'.format(im_id))

    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_600ms/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_600ms/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_600ms/{}.png'.format(im_id))

    save_dir = 'results/thingseeg2_preproc/sub10/versatile_diffusion_800ms/'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    x[0].save(save_dir + '{}.png'.format(im_id))
        
    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_200ms_1regressor/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_200ms_1regressor/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_200ms_1regressor/{}.png'.format(im_id))

    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_400ms_1regressor/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_400ms_1regressor/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_400ms_1regressor/{}.png'.format(im_id))

    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_600ms_1regressor/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_600ms_1regressor/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_600ms_1regressor/{}.png'.format(im_id))

    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_800ms_1regressor/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_800ms_1regressor/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_800ms_1regressor/{}.png'.format(im_id))
        
        
    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_null/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_null/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_null/{}.png'.format(im_id))
        
    # if not osp.exists('results/thingseeg2_preproc/versatile_diffusion_800ms_dnnvision/'):
    #     os.makedirs('results/thingseeg2_preproc/versatile_diffusion_800ms_dnnvision/')
    # x[0].save('results/thingseeg2_preproc/versatile_diffusion_800ms_dnnvision/{}.png'.format(im_id))
        
    # if not osp.exists('results/thingseeg2/versatile_diffusion_avg1_200ms/'):
    #     os.makedirs('results/thingseeg2/versatile_diffusion_avg1_200ms/')
    # x[0].save('results/thingseeg2/versatile_diffusion_avg1_200ms/{}.png'.format(im_id))
        
    # if not osp.exists('results/thingseeg2/versatile_diffusion_avg1_400ms/'):
    #     os.makedirs('results/thingseeg2/versatile_diffusion_avg1_400ms/')
    # x[0].save('results/thingseeg2/versatile_diffusion_avg1_400ms/{}.png'.format(im_id))
        
    # if not osp.exists('results/thingseeg2/versatile_diffusion_avg1_600ms/'):
    #     os.makedirs('results/thingseeg2/versatile_diffusion_avg1_600ms/')
    # x[0].save('results/thingseeg2/versatile_diffusion_avg1_600ms/{}.png'.format(im_id))
        
    # if not osp.exists('results/thingseeg2/versatile_diffusion_avg1_800ms/'):
    #     os.makedirs('results/thingseeg2/versatile_diffusion_avg1_800ms/')
    # x[0].save('results/thingseeg2/versatile_diffusion_avg1_800ms/{}.png'.format(im_id))
      

