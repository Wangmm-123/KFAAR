import os
import os.path as osp
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from sklearn import metrics
from facenet_pytorch import MTCNN, InceptionResnetV1
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time

from new_data import LFW
from model.projector import Projector
import model.recognizer as sphere
from model.arcface import ResNet
from utils.utils import gen_batch_two_keys, tensor2imgpath, gen_batch_key, softmax_temperature
import dnnlib
import legacy
import hopenet
import torchvision
from argparse import Namespace
from models.psp import pSp
from model.face_vid2vid.driven_demo import init_facevid2vid_pretrained_model, drive_source_demo


def run_on_batch(inputs, net):
    """将输入图像传入pSp/e4e模型，返回图像和latents"""
    images, latents = net(inputs.float(), randomize_noise=False, return_latents=True)
    return images, latents


def mapping(G, p_z, z):
    """风格混合：融合projector输出的latent和e4e的latent"""
    ws = z
    p_ws = G.mapping(p_z, None)
    col_styles = [0, 1, 2, 3]
    rest_styles = [4,5,6, 7, 8, 9, 10, 11, 12, 13]
    mixed_w = []
    for out_w, in_w in zip(p_ws, ws):
        w = out_w.clone()
        w[col_styles] = in_w[col_styles]
        w[rest_styles] = torch.lerp(in_w[rest_styles], w[rest_styles], 0.6)
        mixed_w.append(w)
    mixed_w = torch.stack(mixed_w, dim=0)
    return p_ws, mixed_w


@torch.no_grad()
def generate_virtual_dataset_normal(args):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    ckpt = args.ckpt
    batch_size = args.batch_size
    device = args.device
    ndigit = args.ndigit

    virtual_root = 'new_virtual/'


    if osp.isdir(virtual_root):
        os.system(f'rm -r {virtual_root}')
        os.rmdir(virtual_root)
        os.mkdir(virtual_root)
    else:
        os.mkdir(virtual_root)


    test_set = LFW(mode='test')
    test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=True)

    encoder = InceptionResnetV1(pretrained='vggface2').to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    projector = Projector(ndigit=ndigit).to(device)
    projector.load_state_dict(torch.load(ckpt))
    projector.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    model_path = 'pretrained_models/e4e_ffhq_encode_256.pt'
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts).to(device)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False


    network_pkl = 'final_models/celeba256.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        generator = legacy.load_network_pkl(f)['G_ema'].to(device)
    generator.eval()
    for param in generator.parameters():
        param.requires_grad = False


    face_vid2vid_cfg = "pretrained_models/facevid2vid/vox-256.yaml"
    face_vid2vid_ckpt = "pretrained_models/facevid2vid/00000189-checkpoint.pth.tar"
    g1, kp_detector, he_estimator, estimate_jacobian = init_facevid2vid_pretrained_model(face_vid2vid_cfg,
                                                                                         face_vid2vid_ckpt)
    g1.eval()
    kp_detector.eval()
    he_estimator.eval()

    for s, batch in enumerate(tqdm(test_loader)):
        x1, x2, y = batch
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        k, _= gen_batch_two_keys(batch_size, ndigit)

        _, latents_x1 = run_on_batch(x1, net)
        _, latents_x2 = run_on_batch(x2, net)
        _, latents_y = run_on_batch(y, net)
        z_x1 = latents_x1.to(device)
        z_x2 = latents_x2.to(device)
        z_y = latents_y.to(device)

        # -------------------------- 1. Projector前向传播与损失计算 --------------------------
        # Projector生成latent
        z_x1k = projector(encoder(x1), k)
        z_x2k = projector(encoder(x2), k)
        z_yk = projector(encoder(y), k)

        # mapping + StyleGAN生成图像
        _, w_x1k = mapping(generator, z_x1k, z_x1)
        _, w_x2k = mapping(generator, z_x2k, z_x2)
        _, w_yk = mapping(generator, z_yk, z_y)

        x1k = generator.synthesis(w_x1k, noise_mode='const')
        x2k = generator.synthesis(w_x2k, noise_mode='const')
        yk = generator.synthesis(w_yk, noise_mode='const')

        x1k_p = torch.empty(batch_size, 3, 256, 256).to(device)
        x2k_p = torch.empty(batch_size, 3, 256, 256).to(device)
        yk_p = torch.empty(batch_size, 3, 256, 256).to(device)
        for i in range(batch_size):
            source_x1k = x1k[i] * torch.tensor(std).view(3, 1, 1).to(device) + torch.tensor(mean).view(3, 1, 1).to(
                device)
            source_x2k = x2k[i] * torch.tensor(std).view(3, 1, 1).to(device) + torch.tensor(mean).view(3, 1, 1).to(
                device)
            source_yk = yk[i] * torch.tensor(std).view(3, 1, 1).to(device) + torch.tensor(mean).view(3, 1, 1).to(device)
            target_x1 = x1[i] * torch.tensor(std).view(3, 1, 1).to(device) + torch.tensor(mean).view(3, 1, 1).to(device)
            target_x2 = x2[i] * torch.tensor(std).view(3, 1, 1).to(device) + torch.tensor(mean).view(3, 1, 1).to(device)
            target_y = y[i] * torch.tensor(std).view(3, 1, 1).to(device) + torch.tensor(mean).view(3, 1, 1).to(device)
            x1k_p[i] = transforms.Normalize(mean=mean, std=std)(
                drive_source_demo(source_x1k, target_x1, g1, kp_detector, he_estimator, estimate_jacobian) / 255.0)
            x2k_p[i] = transforms.Normalize(mean=mean, std=std)(
                drive_source_demo(source_x2k, target_x2, g1, kp_detector, he_estimator, estimate_jacobian) / 255.0)
            yk_p[i] = transforms.Normalize(mean=mean, std=std)(
                drive_source_demo(source_yk, target_y, g1, kp_detector, he_estimator, estimate_jacobian) / 255.0)

        for j in range(batch_size):
            dir1 = osp.join(virtual_root, str(s * 2 * batch_size + 2 * j))
            dir2 = osp.join(virtual_root, str(s * 2 * batch_size + 2 * j + 1))
            os.mkdir(dir1)
            os.mkdir(dir2)

            tensor2imgpath(x1k[j], osp.join(dir1, 'x1k.png'))
            tensor2imgpath(x2k[j], osp.join(dir1, 'x2k.png'))
            tensor2imgpath(x1[j], osp.join(dir1, 'x1_ori.png'))
            tensor2imgpath(x2[j], osp.join(dir1, 'x2_ori.png'))
            tensor2imgpath(x1k_p[j], osp.join(dir1, 'x1k_p.png'))
            tensor2imgpath(x2k_p[j], osp.join(dir1, 'x2k_p.png'))

            tensor2imgpath(yk[j], osp.join(dir2, 'yk.png'))
            tensor2imgpath(y[j], osp.join(dir2, 'y_ori.png'))
            tensor2imgpath(yk_p[j], osp.join(dir2, 'yk_p.png'))

if __name__ == '__main__':
    parser = ArgumentParser(description='Train an encoder for predicting latents of the virtual faces.')
    parser.add_argument('-c', '--ckpt', type=str, default='2-5/best_projector.pt')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-n', '--ndigit', type=int, default=8)
    parser.add_argument('-s', '--seed', type=int, default=41)
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generate_virtual_dataset_normal(args)
