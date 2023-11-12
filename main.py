import cv2 
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# Create Flask app
app = Flask(__name__)

# Load model from file model.keras
model = tf.keras.models.load_model('model.keras')
model.make_predict_function()

# Define classes
classes = [
  "cloud",
  "sun",
  "pants",
  "umbrella",
  "table",
  "ladder"
  "eyeglasses",
  "clock",
  "scissors",
  "cup",
]

# Create route for index.html
@app.route('/')
def index():
    return render_template('index.html')

# Create route for recognize
@app.route('/recognize', methods =['POST'])
def recognize():
    if request.method == 'POST':
        # Get image from request
        data = request.get_json()
        imageBase64 = data['image']
        imgBytes = base64.b64decode(imageBase64)

        # Save image to temp.jpg
        with open("temp.jpg", "wb") as temp:
            temp.write(imgBytes)

        # Read image and resize to 28x28
        image = cv2.imread('temp.jpg')
        image = cv2.resize(image,(28,28), interpolation=cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image_prediction = np.reshape(image_gray,(28,28,1))
        image_prediction = (255 - image_prediction.astype('float')) / 255

        # Get prediction by 
        prediction = model.predict(np.expand_dims(image_prediction, axis=0))[0]
        prediction = classes[np.argmax(prediction, axis=0)]

        return jsonify({
            'prediction': str(prediction),
            'status': True
        })
        
if __name__ == '__main__':
    app.run(debug = True)
    