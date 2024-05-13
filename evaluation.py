import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import tensorflow as tf
import tensorflow.keras.models as keras_models
# Assuming TensorFlow is not strictly required
# If needed, import relevant functions from TensorFlow
import os

data_dir = 'data/lfw/lfw'
pairs_path = 'data/lfw/pairs.txt'

batch_size = 16
epochs = 15
workers = 0 if os.name == 'nt' else 8
# Assuming the script is in the project's root directory
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

orig_img_ds = datasets.ImageFolder(data_dir, transform=None)


orig_img_ds.samples = [
    (p, p)
    for p, _ in orig_img_ds.samples
]

loader = DataLoader(
    orig_img_ds,
    num_workers=workers,
    batch_size=batch_size,
)

crop_paths = []
box_probs = []

for i, (x, b_paths) in enumerate(loader):
    crops = [p.replace(data_dir, data_dir ) for p in b_paths]
    crop_paths.extend(crops)
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')


trans = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(data_dir, transform=trans)

embed_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SequentialSampler(dataset)
)

# Your absolute paths (replace with actual locations)
model_json_path = 'C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/keras-facenet-h5/model.json'
model_weights_path = 'C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/keras-facenet-h5/model.h5'

# Open the JSON file
with open(model_json_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the model architecture from JSON
model = keras_models.model_from_json(loaded_model_json)

# Load the model weights
model.load_weights(model_weights_path)

FRmodel = model

def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)

classes = []
embeddings = []
FRmodel.eval()  # Set the model to evaluation mode (if applicable)
with torch.no_grad():
    for xb, yb in embed_loader:
        xb = xb.to(device)  # Assuming device is defined (e.g., GPU)
        b_embeddings = img_to_encoding(xb, FRmodel)  # Extract first image
        b_embeddings = b_embeddings.to('cpu').numpy()
        classes.extend(yb.numpy())
        embeddings.extend(b_embeddings)

embeddings_dict = dict(zip(crop_paths, embeddings))
