from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/train/ants/460874319_0a45ab4d05.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("Totensor", img_tensor)


# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("resize", img_resize,0)

# compose
trans_compose = transforms.Compose([trans_resize,trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize", img_resize_2, 1)

# randomcrop
trans_randomcrop = transforms.RandomCrop((300,200))
trans_compose_2 = transforms.Compose([trans_randomcrop, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("randomHW", img_crop, i)

writer.close()