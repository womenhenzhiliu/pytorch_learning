from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# transforms.ToTensor
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)


writer = SummaryWriter("logs")


print(type(img))
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)

writer.add_image('Tensor_image', tensor_img,)
writer.close()