
import base64
from pyexpat import model
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

def get_model():
    global model
    model = load_model("waste_prediction_model.h5")
    print(" * Model loaded!")

# decode the image embedded in the request
def decode_request(req):
    encoded = req["image"]
    decoded = base64.b64decode(encoded)
    return decoded

# preprocess image before sending it to the model
def preprocess(decoded):
    #resize and convert to RGB in case image is in RGBA.
    pil_image = Image.open(io.BytesIO(decoded)).resize((180,180), Image.LANCZOS).convert("RGB") 
    image = np.asarray(pil_image)
    batch = np.expand_dims(image, axis=0)
    return batch

# Function to categorise the prediction:
def categorise(prediction):
  if prediction == 0:
    category = "cardboard"
  if prediction == 1:
    category = "glass"
  if prediction == 2:
    category = "metal"
  if prediction == 3:
    category = "paper"
  if prediction == 4:
    category = "plastic"
  if prediction == 5:
    category = "trash"                
  return category

print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    print("[+] request received")

    # Get the data from the request and convert to correct format:
    req = request.get_json(force=True)
    image = decode_request(req)
    processed_image = preprocess(image)

    # Prediction by the model
    prediction = np.argmax(model.predict(processed_image), axis = -1)
    prediction = 1
    predicted_category = categorise(prediction)
    response = {"prediction": predicted_category}
    return jsonify(response) # Return prediction in json format.
