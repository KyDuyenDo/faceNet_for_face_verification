from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

from PIL import Image
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
    image_size=160,
    margin=14,
    device=device,
    selection_method='center_weighted_size'
)
img = Image.open("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/diem/01.jpg")
img1 = Image.open("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/diem/01.jpg")
img2 = Image.open("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/diem/02.jpg")
img3 = Image.open("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/diem/03.jpg")
img4 = Image.open("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/diem/04.jpg")
img5 = Image.open("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/diem/05.jpg")

img_cropped = mtcnn(img)
img_cropped1 = mtcnn(img1)
img_cropped2 = mtcnn(img2)
img_cropped3 = mtcnn(img3)
img_cropped4 = mtcnn(img4)
img_cropped5 = mtcnn(img5)

# from torchvision import transforms

# # Define transformations
# transform = transforms.Compose([
#     np.float32,
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])

# Preprocess images
# img = transform(img)
# img1 = transform(img1)
# img2 = transform(img2)
# img3 = transform(img3)
# img4 = transform(img4)
# img5 = transform(img5)
# Add an extra dimension for batch size (even though it's a single image)
img_embedding = resnet(img_cropped.unsqueeze(0)).detach()
img_embedding1 = resnet(img_cropped1.unsqueeze(0)).detach()
img_embedding2 = resnet(img_cropped2.unsqueeze(0)).detach()
img_embedding3 = resnet(img_cropped3.unsqueeze(0)).detach()
img_embedding4 = resnet(img_cropped4.unsqueeze(0)).detach()
img_embedding5 = resnet(img_cropped5.unsqueeze(0)).detach()

dist1 = np.linalg.norm(img_embedding1 - img_embedding)
dist2 = np.linalg.norm(img_embedding2 - img_embedding)
dist3 = np.linalg.norm(img_embedding3 - img_embedding)
dist4 = np.linalg.norm(img_embedding4 - img_embedding)
dist5 = np.linalg.norm(img_embedding5 - img_embedding)
print(dist1)
print(dist2)
print(dist3)
print(dist4)
print(dist5)