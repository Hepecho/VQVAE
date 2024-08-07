import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """相同channel数的残差块"""
    def __init__(self, dim):
        """
        HW不变且中间通道数和输入输出通道数相同的残差块，单层
        :param dim: 通道数
        """
        super(ResidualBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)  # 保证输入输出HW相同，默认通道数也相同
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)  # 保证输入输出HW相同

    def forward(self, x):
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return x + out


class VQVAE(nn.Module):
    def __init__(self, in_dim, h_dim, n_e):
        """
        VQ-VAE初始化
        :param in_dim: 输入image维度
        :param h_dim: 中间hidden code维度 / VQ码本向量维度
        :param n_e: VQ码本中所有离散向量的数量
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, 4, 2, 1), nn.ReLU(),  # 2倍下采样
            nn.Conv2d(h_dim // 2, h_dim, 4, 2, 1), nn.ReLU(),  # 2倍下采样
            nn.Conv2d(h_dim, h_dim, 3, 1, 1),  # 3*3卷积，不改变HW
            ResidualBlock(h_dim), ResidualBlock(h_dim)  # 两层3*3残差块
        )
        self.vq_embedding = nn.Embedding(n_e, h_dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
        # 离散化VQ码本实际上是一个embedding层
        self.decoder = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, 3, 1, 1),  # 这里转置卷积等效于卷积操作
            ResidualBlock(h_dim), ResidualBlock(h_dim),
            nn.ConvTranspose2d(h_dim, h_dim // 2, 4, 2,1),  # 转置卷积进行上采样，HW扩大2倍
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, in_dim, 4, 2, 1)
        )
        self.num_down_sample = 2

    def forward(self, x):
        ze = self.encoder(x)
        # 最近邻离散化
        # 目的：将encoder输出向量z映射为离散的one-hot向量，它索引指向与z最近邻的码本向量e_j，得到最终输出的向量记作z_q
        # 流程：总共有B*H*W个C维的向量要和VQ码本中K个D维的向量求最近邻（我们保证C==D，即e_dim，K即n_e）
        B, C, H, W = ze.shape
        K = self.vq_embedding.weight.shape[0]
        embedding = self.vq_embedding.weight.data
        # 1. 利用广播机制计算欧氏距离矩阵
        ze_broadcast = ze.reshape(B, 1, C, H, W)
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        distance = torch.sum((ze_broadcast - embedding_broadcast)**2, dim=2)  # (B, K, H, W)
        # 2. 获取最近邻的index矩阵
        index = torch.argmin(distance, dim=1)  # (B, H, W)
        zq = self.vq_embedding(index).permute(0, 3, 1, 2)  # (B, H, W, C)
        # stop gradient
        decoder_input = ze + (zq - ze).detach()  # 为了保证bp时zq直接copy给ze，而不传递梯度给embeddding

        # decoder重建输入
        x_hat = self.decoder(decoder_input)
        return ze, zq, x_hat  # 返回ze和zq是为了计算第二和第三项损失

    @torch.no_grad()
    def encode(self, x):
        ze = self.encoder(x)
        B, C, H, W = ze.shape
        K = self.vq_embedding.weight.shape[0]
        embedding = self.vq_embedding.weight.data
        # 1. 利用广播机制计算欧氏距离矩阵
        ze_broadcast = ze.reshape(B, 1, C, H, W)
        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
        distance = torch.sum((ze_broadcast - embedding_broadcast) ** 2, dim=2)  # (B, K, H, W)
        # 2. 获取最近邻的index矩阵
        index = torch.argmin(distance, dim=1)  # (B, H, W)
        return index

    @torch.no_grad()
    def decode(self, discrete_latent):
        zq = self.vq_embedding(discrete_latent).permute(0, 3, 1, 2)
        x_hat = self.decoder(zq)
        return x_hat

    @torch.no_grad()
    def check_codebook(self):
        print(self.vq_embedding.weight.shape)
        print(self.vq_embedding.weight[0])

