import json
import numpy as np
from PIL import Image
import base64
import io
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np


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

def img_to_encoding(image_data_base64, mtcnn, resnet):
    imgdata = base64.b64decode(image_data_base64)
    img = Image.open(io.BytesIO(imgdata)) 
    # img = img.convert('RGB')
    img_cropped = mtcnn(img)
    img_embedding = resnet(img_cropped.unsqueeze(0)).detach()
    return img_embedding


def verify(image_path, identity, mtcnn, resnet):
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above
    encoding = img_to_encoding(image_path, mtcnn, resnet)
    identity_image = img_to_encoding(identity ,mtcnn, resnet)
    # Step 2: Compute distance with identity's image
    dist = np.linalg.norm(encoding - identity_image)
    
    # Step 3: Open the door if dist < 0.7, else don't open
    if dist < 0.7:
        print("the same person!, the distance is " + str(dist))
        door_open = True
    else:
        print("two different people!, the distance is " + str(dist))
        door_open = False

    return door_open
 
# SERVER
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

# API endpoint to multiply images and predict
@app.route('/predict', methods=['POST'])
def predict_api():
    if request.method == 'POST':
        try:
            data = json.loads(request.data)
            image1_data = data['image1']
            image2_data = data['image2']
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON data: {str(e)}'}), 400
        
        result = verify(image1_data, image2_data, mtcnn, resnet)

        return jsonify({
            'prediction': result,  # Replace with actual prediction
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)