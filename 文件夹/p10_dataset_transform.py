import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./CIFAR10dataset", train=True, transform=dataset_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10dataset", train=False, transform=dataset_transform,download=True)

# print(train_set[0])
# print(test_set.classes)
#
# img,label = train_set[0]
# print(test_set.classes[label])
# img.show(img)

# print(train_set[0])

writer = SummaryWriter("p10logs")
for i in range(10):
    img,label = train_set[i]
    writer.add_image("cifar10", img, i)

writer.close()
