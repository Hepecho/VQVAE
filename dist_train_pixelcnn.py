import time
import os
import torch
import torch.nn as nn
from runx.logx import logx
from dataset import get_dataloader
from model import VQVAE
from utils import *


def train_generative_model(vqvae: VQVAE,
                           model,
                           img_shape=None,
                           num_workers=1,
                           device='cuda',
                           ckpt_dir='checkpoints/',
                           batch_size=64,
                           lr=1e-3,
                           n_epochs=50):
    model.to(device)
    model.train()
    vqvae.to(device)
    vqvae.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    ce_loss = nn.CrossEntropyLoss()
    data_iterator = get_dataloader(batch_size=batch_size, img_shape=img_shape, num_workers=num_workers)
    for epoch in range(n_epochs):
        start_time = time.time()
        total_loss = 0
        for batch in iter(data_iterator):
            batch_img, _ = batch
            batch_img = batch_img.to(device)
            with torch.no_grad():
                x = vqvae.encode(batch_img)  # 得到离散变量的index
            predict_x = model(x)  # pixelcnn通过mask自回归式预测下一个index
            loss = ce_loss(predict_x, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_img)
        total_loss /= len(data_iterator.dataset)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        if (epoch + 1) % 20 == 0 or epoch == n_epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f'pixelcnn_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), ckpt_path)
        logx.msg(f'Epoch: {epoch} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logx.msg(f'Train Loss: {total_loss:.4f}')


if __name__ == '__main__':
    pass
