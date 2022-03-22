

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./CIFAR10dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3)

    def forward(self,x):
        x = self.conv1(x)
        return x

Model = model()
writer = SummaryWriter(log_dir="logs")
step = 0
for data in dataloader:
    imgs, labels = data
    output = Model(imgs)
    print(output.shape)
    writer.add_images(tag="conv2d_input", img_tensor=imgs, global_step=step )

    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images(tag="conv2d_output", img_tensor=output,global_step=step)
    step += 1


