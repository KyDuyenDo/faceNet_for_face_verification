from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import numpy as np
import tensorflow as tf


import tensorflow.keras.models as keras_models

# Assuming the script is in the project's root directory

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

# Now you can use the loaded model for inference or training
# print(model.inputs)
# print(model.outputs)

FRmodel = model

#tf.keras.backend.set_image_data_format('channels_last')
def img_to_encoding(image_path, model):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
    img = np.around(np.array(img) / 255.0, decimals=12)
    x_train = np.expand_dims(img, axis=0)
    embedding = model.predict_on_batch(x_train)
    return embedding / np.linalg.norm(embedding, ord=2)


database = {}
# database["danielle"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/danielle.png", FRmodel)
# database["younes"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/younes.jpg", FRmodel)
# database["tian"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/tian.jpg", FRmodel)
# database["andrew"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/andrew.jpg", FRmodel)
# database["kian"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/kian.jpg", FRmodel)
# database["dan"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/dan.jpg", FRmodel)
# database["sebastiano"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/sebastiano.jpg", FRmodel)
# database["bertrand"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/bertrand.jpg", FRmodel)
# database["kevin"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/kevin.jpg", FRmodel)
# database["felix"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/felix.jpg", FRmodel)
# database["benoit"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/benoit.jpg", FRmodel)
# database["arnaud"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/arnaud.jpg", FRmodel)
# database["diem"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/diem.png", FRmodel)
database["duyen"] = img_to_encoding("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/diem/01.jpg", FRmodel)

def verify(image_path, identity, database, model):
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above
    encoding = img_to_encoding(image_path, model)
    
    # Step 2: Compute distance with identity's image
    dist = np.linalg.norm(encoding - database[identity])
    
    # Step 3: Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome in!, the distance is " + str(dist))
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away, the distance is " + str(dist))
        door_open = False

    return dist, door_open

verify("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/duyen/01.jpg", "duyen", database, FRmodel)
verify("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/duyen/02.jpg", "duyen", database, FRmodel)
verify("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/duyen/03.jpg", "duyen", database, FRmodel)
verify("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/duyen/04.jpg", "duyen", database, FRmodel)
verify("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/duyen/05.jpg", "duyen", database, FRmodel)
# verify("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/images/diem.png", "duyen", database, FRmodel)
# verify("C:/Users/Admin/Desktop/facenet-face-verification-and-face-recognition-main/camera_duyen.jpg", "duyen", database, FRmodel)