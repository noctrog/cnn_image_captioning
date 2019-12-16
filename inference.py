import argparse

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import model

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True, help='Input image')
args = parser.parse_args()

inference_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open(args.i)
image_r = image.resize((224, 224))
image_v = inference_transforms(image_r).cuda()
image_v = image_v.view(1, *image_v.shape)

dicts = torch.load('./weights/dicts')
stoi = dicts['stoi']
itos = dicts['itos']

cnn = model.CNN_CNN_HA_CE(len(stoi), 300, n_layers=20, max_length=15).cuda()
cnn.load()
cnn.eval()

label = cnn.sample(image_v, stoi, itos)[0]

plt.imshow(image)
plt.title(label)
plt.show()
