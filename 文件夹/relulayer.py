import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader

# input = torch.Tensor([[1,-0.5],
#                       [-1,3]])
#
# input = torch.reshape(input,(-1,1,2,2))
#
#
# print(input.shape)
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./CIFAR10dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(dataset=dataset,batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output

model = Model()

# output = model(input)
#
# print(output)

writer = SummaryWriter(log_dir="sigmoid_logs")
step = 0

for data in dataloader:
    imgs, labels = data
    writer.add_images(tag="outputsigmoid",img_tensor=model(imgs),global_step=step)
    writer.add_images(tag="inputsigmoid",img_tensor=imgs, global_step=step)
    step += 1

writer.close()


