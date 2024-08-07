from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


def download_mnist():
    """下载MINIST数据集，查看数据集格式"""
    # 下载数据集
    train_set = datasets.MNIST("data/", train=True)

    # test_set = datasets.MNIST("data/", train=False, download=True)
    print(len(train_set))
    img, label = train_set[0]
    img = transforms.ToTensor()(img)
    print(img.shape)  # torch.Size([1, 28, 28])

    # 加载数据
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


class MNISTImageDataset(Dataset):
    def __init__(self, root='data/', train=True, transform=None):
        self.dataset = datasets.MNIST(root, train=train)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataset[index]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataset)


def get_dataloader(batch_size,
                   img_shape=None,
                   num_workers=4,
                   root='data/',
                   train=True,
                   **kwargs):
    if img_shape is None:
        pipeline = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNISTImageDataset(root=root, transform=pipeline, train=train)
    else:
        pipeline = transforms.Compose([
            transforms.Resize(img_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNISTImageDataset(root=root, transform=pipeline, train=train)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


if __name__ == '__main__':
    download_mnist()