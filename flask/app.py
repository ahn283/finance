import io
import json

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

# let's define a function that reads in images data, preprocessing it, passes it into our model, and returns the image class.
import json
imagenet_class_index = json.load(
    open('./imagenet_class_index.json')
)

model = models.vgg16(pretrained=True)

image_transforms = transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(
         [0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225,]
     )]
)

def get_prediction(image_bytes):
    # get_prediction() function creates a PIL image object based on the received bytes
    image = Image.open(io.BytesIO(image_bytes))
    # apply the required image transfomrs to create an image tensor
    tensor = image_transforms(image)
    # we perform the forward pass and find the class with highest probability y
    outputs = model(tensor)
    _, y = outputs.max(1)
    predicted_idx = str(y.item())
    # we look up the class name using the output class value
    return imagenet_class_index[predicted_idx]

# web server object is called app, created but not running yet
app = Flask(__name__)
# we set our endpoint to '/predict' and configured it to handle POST requests.
@app.route('/predict', method=['POST'])

def predict():
    # read the image, get the prediction, and returns the image class in JSON format.
    if request.method == 'POST':
        file = request.files['file']
    
    img_bytes = file.read()
    class_id, class_name = get_prediction(image_bytes=img_bytes)
    return jsonify({'class_id': class_id,
                    'class_name': class_name})
    
# we need to add the code so that the web server runs when we execute app.py
if __name__ == '__main__':
    app.run()