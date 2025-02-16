from datetime import datetime
from ldm_inverse.condition_methods import get_conditioning_method
from ldm.models.diffusion.diffstategrad_ddim import DDIMSampler
from data.dataloader import get_dataset, get_dataloader
from scripts.utils import clear_color, mask_generator
import matplotlib.pyplot as plt
from ldm_inverse.measurements import get_noise, get_operator
from functools import partial
import numpy as np
from model_loader import load_model_from_config, load_yaml
import os
import torch
import torchvision.transforms as transforms
import argparse
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
from pytorch_lightning import seed_everything

def compute_nmse(y_true_temp, y_pred_temp, eps=1e-6):
    if isinstance(y_true_temp, np.ndarray):
        y_true = torch.from_numpy(y_true_temp)
    else:
        y_true = y_true_temp.clone()
    
    if isinstance(y_pred_temp, np.ndarray):
        y_pred = torch.from_numpy(y_pred_temp)
    else:
        y_pred = y_pred_temp.clone()

    device = y_true.device
    y_pred = y_pred.to(device)

    # Reshape from (256, 256, 3) to (3, 256, 256)
    if y_true.shape[-1] == 3:
        y_true = y_true.permute(2, 0, 1)
        y_pred = y_pred.permute(2, 0, 1)

    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    loss = torch.mean(torch.pow(y_true_flat - y_pred_flat, 2), dim=-1) / (
        torch.mean(torch.pow(y_true_flat, 2), dim=-1) + eps
    )
    return torch.mean(loss)

def compute_psnr(y_true_temp, y_pred_temp, max_val=1.0, eps=1e-6):
    if isinstance(y_true_temp, np.ndarray):
        y_true = torch.from_numpy(y_true_temp)
    else:
        y_true = y_true_temp.clone()
    
    if isinstance(y_pred_temp, np.ndarray):
        y_pred = torch.from_numpy(y_pred_temp)
    else:
        y_pred = y_pred_temp.clone()

    device = y_true.device
    y_pred = y_pred.to(device)

    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

    # Clamp the values to the range [0, max_val]
    y_true = torch.clamp(y_true, 0, max_val)
    y_pred = torch.clamp(y_pred, 0, max_val)

    msef = torch.mean(torch.pow(y_true_flat - y_pred_flat, 2), dim=-1)
    maxf = torch.amax(y_true_flat, dim=-1)
    psnr = 20 * torch.log10(maxf) - 10 * torch.log10(msef)
    return torch.mean(psnr)

def compute_ssim(y_true_temp, y_pred_temp):
    y_true_np = y_true_temp
    y_pred_np = y_pred_temp
    meas_ssim, diff = ssim(y_true_np, y_pred_np, full=True, multichannel=True, channel_axis=2, data_range=1.0)
    return meas_ssim

def compute_lpips(y_true_temp, y_pred_temp):
    if isinstance(y_true_temp, np.ndarray):
        y_true = torch.from_numpy(y_true_temp)
    else:
        y_true = y_true_temp.clone()
    
    if isinstance(y_pred_temp, np.ndarray):
        y_pred = torch.from_numpy(y_pred_temp)
    else:
        y_pred = y_pred_temp.clone()

    # Reshape from (256, 256, 3) to (3, 256, 256)
    if y_true.shape[-1] == 3:
        y_true = y_true.permute(2, 0, 1)
        y_pred = y_pred.permute(2, 0, 1)

    device = y_true.device
    y_pred = y_pred.to(device)
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    lpips_score = loss_fn(y_true, y_pred)
    return lpips_score.item()

def get_model(args):
    config = OmegaConf.load(args.ldm_config)
    model = load_model_from_config(config, args.diffusion_config)

    return model

def make_folder(sample_path, opt):
    # Get current date and time
    current_datetime = datetime.now()
    # Format date and time as string
    date_time_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    # Create folder with current date and time as the name
    folder_name = date_time_str

    # Adding omega and gamma hyperparameters
    variance_cutoff_str = "-var_cutoff=({:1})".format(opt.var_cutoff)
    image_str = "-file_id=({})".format(opt.image_id)
    pixel_lr_str = "-pixel_lr=({:.0e})".format(opt.pixel_lr)
    latent_lr_str = "-latent_lr=({:.0e})".format(opt.latent_lr)
    period_str = "-period=({:d})".format(opt.period)

    folder_name += variance_cutoff_str
    folder_name += image_str
    folder_name += pixel_lr_str
    folder_name += latent_lr_str
    folder_name += period_str

    folder_path = os.path.join(sample_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
    return folder_path

parser = argparse.ArgumentParser()
parser.add_argument('--model_config', type=str)
parser.add_argument('--ldm_config', default="configs/latent-diffusion/ffhq-ldm-vq-4.yaml", type=str)
parser.add_argument('--diffusion_config', default="models/ldm/model.ckpt", type=str)
parser.add_argument('--task_config', default="configs/tasks/gaussian_deblur_config.yaml", type=str)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--ddim_steps', default=500, type=int)
parser.add_argument('--ddim_eta', default=0.0, type=float)
parser.add_argument('--n_samples_per_class', default=1, type=int)
parser.add_argument('--ddim_scale', default=1.0, type=float)
parser.add_argument('--image_id', default=60000, type=int)
parser.add_argument('--var_cutoff', default=0.99, type=float)
parser.add_argument('--pixel_lr', default=1e-2, type=float)
parser.add_argument('--latent_lr', default=5e-3, type=float)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--period', default=1, type=int)

args = parser.parse_args()

if args.task_config != "configs/tasks/phase_retrieval_config.yaml":
    seed_everything(args.seed)
else:
    print("avoid")

# Load configurations
task_config = load_yaml(args.task_config)

# Device setting
device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
print(f"Device set to {device_str}.")
device = torch.device(device_str)  

# Loading model
model = get_model(args)

sampler = DDIMSampler(model) # Sampling using DDIM

# Prepare Operator and noise
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])
noiser = get_noise(**measure_config['noise'])
print(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

# Prepare conditioning method
cond_config = task_config['conditioning']
cond_method = get_conditioning_method(cond_config['method'], model, operator, noiser, **cond_config['params'])
measurement_cond_fn = cond_method.conditioning
print(f"Conditioning sampler : {task_config['conditioning']['main_sampler']}")

# Instantiating sampler
sample_fn = partial(sampler.posterior_sampler, measurement_cond_fn=measurement_cond_fn, operator_fn=operator.forward,
                                        S=args.ddim_steps,
                                        cond_method=task_config['conditioning']['main_sampler'],
                                        conditioning=None,
                                        ddim_use_original_steps=True,
                                        batch_size=args.n_samples_per_class,
                                        shape=[3, 64, 64], # Dimension of latent space
                                        verbose=False,
                                        unconditional_guidance_scale=args.ddim_scale,
                                        unconditional_conditioning=None, 
                                        eta=args.ddim_eta, pixel_lr=args.pixel_lr, latent_lr=args.latent_lr,
                                        var_cutoff=args.var_cutoff, period=args.period)

# Working directory
out_path = os.path.join(args.save_dir)
spec_path = make_folder(out_path, args)
os.makedirs(spec_path, exist_ok=True)

# Prepare dataloader
data_config = task_config['data']
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )

# Exception) In case of inpainting, we need to generate a mask 
if measure_config['operator']['name'] == 'inpainting':
  mask_gen = mask_generator(**measure_config['mask_opt'])

# Do inference

# Format the image_id into a 5-digit string padded with zeros
file_name = f"{args.image_id:05d}.png"

# Construct the full file path
file_path = os.path.join("ffhq256", file_name)

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Image file {file_path} does not exist.")

# Open the image file
ref_img = Image.open(file_path).convert('RGB')
ref_img = transform(ref_img).unsqueeze(0)

print(f"Inference for image {args.image_id}")
fname = str(args.image_id).zfill(3)
ref_img = ref_img.to(device)

# Exception) In case of inpainting
if measure_config['operator'] ['name'] == 'inpainting':
    mask = mask_gen(ref_img)
    mask = mask[:, 0, :, :].unsqueeze(dim=0)
    operator_fn = partial(operator.forward, mask=mask)
    measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
    sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn, operator_fn=operator_fn)

    # Forward measurement model
    y = operator_fn(ref_img)
    y_n = noiser(y)

else:
    y = operator.forward(ref_img)
    y_n = noiser(y).to(device)

label = clear_color(y_n)
true = clear_color(ref_img)
plt.imsave(os.path.join(spec_path, 'orig.png'), true)
plt.imsave(os.path.join(spec_path, 'measured.png'), label)

# Sampling
samples_ddim, _ = sample_fn(measurement=y_n)

x_samples_ddim = model.decode_first_stage(samples_ddim.detach())

# Post-processing samples
reconstructed = clear_color(x_samples_ddim)

# Saving images
plt.imsave(os.path.join(spec_path, 'reconstructed.png'), reconstructed)

log_stats = open(os.path.join(spec_path, "log_stats.txt"), "w")

meas_psnr_all = psnr(true, reconstructed)
meas_nmse_all = compute_nmse(true, reconstructed)
meas_ssim_all = compute_ssim(true, reconstructed)
meas_lpips_all = compute_lpips(true, reconstructed)

print("measurement psnr: ", meas_psnr_all.item())
print("measurement nmse: ", meas_nmse_all.item())
print("measurement ssim: ", meas_ssim_all)
print("measurement lpips: ", meas_lpips_all)

log_stats.write(f"measurement psnr: {meas_psnr_all} \n")
log_stats.write(f"measurement nmse: {meas_nmse_all} \n")
log_stats.write(f"measurement ssim: {meas_ssim_all} \n")
log_stats.write(f"measurement lpips: {meas_lpips_all} \n")

log_stats.close()
