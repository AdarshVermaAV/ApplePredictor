from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Import for MobileNetV2
import os

app = Flask(__name__)

# Load the model and class names
model = load_model('apple_classifier_mobilenetv.h5')
class_names = np.load('class_names2.npy')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess image
def preprocess_image(image):
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize the image to match model's input shape
    image = image.resize((224, 224))
    image_array = np.array(image)

    # Ensure the image has 3 channels
    if image_array.shape[-1] != 3:
        raise ValueError("Image does not have 3 channels.")

    # Expand dimensions to match the model input
    image_array = np.expand_dims(image_array, axis=0)

    # Preprocess the image for MobileNetV2
    image_array = preprocess_input(image_array)
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
        except Exception as e:
            print(f"Error saving file: {e}")
            return redirect(request.url)

        try:
            # Load and preprocess the image
            image = Image.open(filepath)
            preprocessed_image = preprocess_image(image)

            # Predict
            predictions = model.predict(preprocessed_image)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            return render_template('results.html',
                                   filename=filename,
                                   predicted_class=predicted_class,
                                   confidence=f"{confidence:.2f}%")
        except Exception as e:
            print(f"Error processing image: {e}")
            return redirect(request.url)
    return redirect(request.url)

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=False)
