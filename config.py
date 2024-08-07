from omegaconf import OmegaConf

if __name__ == '__main__':
    config = OmegaConf.load("configs/mnist.yaml")  # 加载模型配置文件
    print(config)
