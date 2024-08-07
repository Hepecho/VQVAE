import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import make_grid
import cv2
from omegaconf import OmegaConf
import argparse
from runx.logx import logx
from model import VQVAE
from pixelcnn_model import PixelCNNWithEmbedding
from dataset import get_dataloader
from dist_train_pixelcnn import train_generative_model
from utils import *


def train_vqvae(model: VQVAE,
                img_shape=None,
                num_workers=1,
                device='cuda',
                ckpt_dir='checkpoints/',
                batch_size=64,
                lr=1e-3,
                n_epochs=100,
                l_w_embedding=1,
                l_w_commitment=0.25):
    # 获取数据集
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    data_iterator = get_dataloader(batch_size=batch_size, img_shape=img_shape, num_workers=num_workers)
    for epoch in range(n_epochs):
        start_time = time.time()
        total_loss = 0
        for batch in iter(data_iterator):
            batch_img, _ = batch
            batch_img = batch_img.to(device)
            ze, zq, x_hat = model(batch_img)
            reconstruct_loss = mse_loss(x_hat, batch_img)
            codebook_loss = l_w_embedding * mse_loss(ze.detach(), zq)
            # zq的梯度最终是给vq_embedding，而且decoder_input专门detach了zq的部分，所以该项只训练vq_embedding
            commitment_loss = l_w_commitment * mse_loss(ze, zq.detach())
            loss = reconstruct_loss + codebook_loss + commitment_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_img)
        total_loss /= len(data_iterator.dataset)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_path)
        logx.msg(f'Epoch: {epoch} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logx.msg(f'Train Loss: {total_loss:.4f}')


def reconstruct(model, x, device, output_dir='outputs/'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        _, _, x_hat = model(x)

    x_hat = x_hat.squeeze(0).cpu()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    x_hat = x_hat.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    # x_hat = cv2.cvtColor(x_hat, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, 'reconstruct.png')
    cv2.imwrite(output_path, x_hat)
    x = x.squeeze(0).cpu()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    x = x.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    input_path = os.path.join(output_dir, 'source.png')
    cv2.imwrite(input_path, x)


def sample_imgs(vqvae: VQVAE,
                gen_model,
                img_shape,
                n_sample=81,
                n_rows=9,
                device='cuda',
                output_dir='outputs/'):
    vqvae.to(device)
    gen_model.to(device)
    vqvae.eval()
    gen_model.eval()
    H = img_shape[0] // (2**vqvae.num_down_sample)
    W = img_shape[1] // (2**vqvae.num_down_sample)
    x = torch.zeros(n_sample, H, W, dtype=torch.long).to(device)  # (n_sample, H', W')  实际上就是e_j的index矩阵

    for i in range(H):
        for j in range(W):
            with torch.no_grad():
                predict_x = gen_model(x)
                # print(predict_x.shape)  # (n_sample, n_e, H', W')  这里第1维的n_e就是color_level，是codebook的概率分布
                # 当前预测位置的离散index的概率分布
                prob_dist = predict_x[:, :, i, j]
                # 归一化
                prob_dist = F.softmax(prob_dist, dim=-1)
                # 不是取概率最大的像素，而是从概率分布中采样
                # torch.multinomial会对input每一行，取num_samples个值，输出的张量是每一次取值时input张量对应行的下标
                pixel = torch.multinomial(input=prob_dist, num_samples=1).to(torch.long)  # (n_sample, 1)
                x[:, i, j] = pixel[:, 0]

    x_hat = vqvae.decode(x)

    # 绘制grid网格
    grid = make_grid(x_hat, nrow=n_rows)  # 设置每行的图片数量，返回一张网格图像

    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().type(torch.uint8).numpy()
    cv2.imwrite(os.path.join(output_dir, 'samples_grid.png'), grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BadNet')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--config', type=str, default='configs/mnist.yaml')
    parser.add_argument('--log', type=str, default='log/')
    parser.add_argument(
        '--action', type=int, default=0,
        help="0 = 'raw' mode"
    )
    args = parser.parse_args()
    logx.initialize(logdir=args.log, coolname=False, tensorboard=False)
    logx.msg(str(args))
    config = OmegaConf.load(args.config)  # 加载模型配置文件
    in_dim = config.img_shape[0]
    h_dim = config.h_dim
    n_e = config.n_e
    # 1. Train VQVAE
    vqvae = VQVAE(in_dim=in_dim, h_dim=h_dim, n_e=n_e)
    if config.vqvae_path == '':
        train_vqvae(model=vqvae,
                    img_shape=(config.img_shape[1], config.img_shape[2]),
                    num_workers=config.num_workers,
                    device=config.device,
                    ckpt_dir=config.ckpt_dir,
                    batch_size=config.batch_size,
                    lr=config.lr,
                    n_epochs=config.n_epochs,
                    l_w_embedding=config.l_w_embedding,
                    l_w_commitment=config.l_w_commitment)
        vqvae.load_state_dict(torch.load(os.path.join(config.ckpt_dir, f'model_epoch_{config.n_epochs}.pth')))
    else:
        vqvae.load_state_dict(torch.load(config.vqvae_path))
    vqvae.check_codebook()
    # 2. Test VQVAE by visualizaing reconstruction result
    dataloader = get_dataloader(16, img_shape=(config.img_shape[1], config.img_shape[2]))
    img = next(iter(dataloader))[0].to(config.device)
    img = img[0].unsqueeze(0)  # (1, 1, 28, 28)
    reconstruct(vqvae, img, config.device, 'outputs/')
    # 3. Train Generative model (Gated PixelCNN in our project)
    gen_model = PixelCNNWithEmbedding(config.pixelcnn.n_blocks,
                                      config.pixelcnn.dim,
                                      config.pixelcnn.linear_dim,
                                      True,
                                      config.n_e)
    if config.pixelcnn.pixelcnn_path == '':
        train_generative_model(vqvae=vqvae,
                               model=gen_model,
                               img_shape=(config.img_shape[1], config.img_shape[2]),
                               num_workers=config.num_workers,
                               device=config.device,
                               ckpt_dir=config.ckpt_dir,
                               batch_size=config.pixelcnn.batch_size,
                               lr=config.pixelcnn.lr,
                               n_epochs=config.pixelcnn.n_epochs)
        gen_model.load_state_dict(torch.load(
            os.path.join(config.ckpt_dir, f'pixelcnn_epoch_{config.pixelcnn.n_epochs}.pth')))
    else:
        gen_model.load_state_dict(torch.load(config.pixelcnn.pixelcnn_path))

    # 4. Sample VQVAE
    sample_imgs(vqvae, gen_model, (config.img_shape[1], config.img_shape[2]),
                81, 9, config.device, 'outputs/')
