from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = tf.keras.models.load_model('Model.h5')

# Define intensity level thresholds
thresholds = {
    'very_low': 0.2,
    'low': 0.4,
    'medium': 0.6,
    'high': 0.8
}

def preprocess_image(file):
    img = Image.open(file.stream)
    img = img.resize((512, 512))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)

def get_intensity(prediction):
    if prediction < thresholds['very_low']:
        return 'Very Low'
    elif prediction < thresholds['low']:
        return 'Low'
    elif prediction < thresholds['medium']:
        return 'Medium'
    else:
        return 'High'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
        except KeyError:
            app.logger.error("No file part in request")
            return render_template('index.html', message='No file part')

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            try:
                img = preprocess_image(file)
                prediction = model.predict(img)[0][0]
                intensity = get_intensity(prediction)
                return render_template('result.html', prediction=prediction, intensity=intensity)
            except Exception as e:
                app.logger.error(f"Error processing image: {e}")
                return render_template('index.html', message='Error processing image')

if __name__ == '__main__':
    app.run(debug=True)
