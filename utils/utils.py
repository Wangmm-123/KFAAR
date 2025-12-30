from typing import Tuple
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from PIL import Image
import functools
import math
import numpy as np
from random import randint
import scipy.io as sio
from .matlab_cp2tform import get_similarity_transform_for_cv2
from model.decoder import Discriminator

def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result
def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def str2key(key: str, device: str = 'cuda') -> torch.Tensor:
    key = torch.tensor([float(digit) for digit in key])
    return key.to(device)


def gen_batch_key(batch_size: int, ndigit: int = 8) -> torch.Tensor:
    keys = []
    for _ in range(batch_size):
        key = ''
        for _ in range(ndigit):
            d = str(randint(0, 1))
            key += d
        key = str2key(key)
        keys.append(key)
    keys = torch.stack(keys, dim=0)
    return keys


def gen_batch_two_keys(batch_size, ndigit: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    keys1 = []
    keys2 = []
    for _ in range(batch_size):
        while True:
            key1 = ''
            for _ in range(ndigit):
                d = str(randint(0, 1))
                key1 += d
            key2 = ''
            for _ in range(ndigit):
                d = str(randint(0, 1))
                key2 += d
            if key1 != key2:
                break
        key1 = str2key(key1)
        key2 = str2key(key2)
        keys1.append(key1)
        keys2.append(key2)
    keys1 = torch.stack(keys1, dim=0)
    keys2 = torch.stack(keys2, dim=0)
    return keys1, keys2




# def gen_batch_key(batch_size: int, ndigit: int = 128) -> torch.Tensor:
#     keys = []
#     for _ in range(batch_size):
#         key = ''
#         for _ in range(ndigit):
#             d = str(randint(0, 1))
#             key += d
#         key = str2key(key)
#         keys.append(key)
#     keys = torch.stack(keys, dim=0)
#     return keys


# def gen_batch_two_keys(batch_size, ndigit: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
#     keys1 = []
#     keys2 = []
#     for _ in range(batch_size):
#         while True:
#             key1 = ''
#             for _ in range(ndigit):
#                 d = str(randint(0, 1))
#                 key1 += d
#             key2 = ''
#             for _ in range(ndigit):
#                 d = str(randint(0, 1))
#                 key2 += d
#             if key1 != key2:
#                 break
#         key1 = str2key(key1)
#         key2 = str2key(key2)
#         keys1.append(key1)
#         keys2.append(key2)
#     keys1 = torch.stack(keys1, dim=0)
#     keys2 = torch.stack(keys2, dim=0)
#     return keys1, keys2


def tensor2image(tensor: torch.Tensor) -> Image.Image:
    out = tensor.clamp(-1, 1).add(1).div(2)
    return transforms.ToPILImage()(out.to('cpu'))


def tensor2imgpath(tensor: torch.Tensor, path: str, nrow: int = None):
    torchvision.utils.save_image(tensor, path, normalize=True, range=(-1, 1), nrow=nrow)


def image2tensor(image: Image.Image, image_size: int = 128) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image)


def imgpath2tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    return image2tensor(image)


def get_discriminator(image_size) -> Discriminator:
    discriminator = Discriminator(from_rgb_activate=True)
    discriminator.forward = functools.partial(discriminator.forward, step=int(math.log2(image_size)) - 2, alpha=0)
    return discriminator


def alignment(src_pts, device: str = 'cuda'):
    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    # crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    s = s / 125. - 1.
    r[:, 0] = r[:, 0] / 48. - 1
    r[:, 1] = r[:, 1] / 56. - 1

    all_tfms = np.empty((s.shape[0], 2, 3), dtype=np.float32)
    for idx in range(s.shape[0]):
        all_tfms[idx, :, :] = get_similarity_transform_for_cv2(r, s[idx, ...])
    all_tfms = torch.from_numpy(all_tfms).to(device)
    return all_tfms


@torch.no_grad()
def process_batch(batch, size: int = 128, device: str = 'cuda'):
    x1, x2, y, lm_x1, lm_x2, lm_y = batch
    theta_x1 = alignment(lm_x1)
    theta_x2 = alignment(lm_x2)
    theta_y = alignment(lm_y)
    grid_x1 = F.affine_grid(theta_x1, size=[x1.shape[0], 3, size, size], align_corners=False)
    grid_x2 = F.affine_grid(theta_x2, size=[x2.shape[0], 3, size, size], align_corners=False)
    grid_y = F.affine_grid(theta_y, size=[y.shape[0], 3, size, size], align_corners=False)
    x1 = F.grid_sample(x1.to(device), grid_x1, align_corners=False)
    x2 = F.grid_sample(x2.to(device), grid_x2, align_corners=False)
    y = F.grid_sample(y.to(device), grid_y, align_corners=False)
    return x1, x2, y


@torch.no_grad()
def crop_batch(batch, net, device: str = 'cuda'):
    imgs = []
    for img in batch:
        img = tensor2image(img)
        img = net(img)
        imgs.append(img)
    out = torch.stack(imgs, dim=0).to(device)
    return out
