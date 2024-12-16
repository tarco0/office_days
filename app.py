from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB upload limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, and JPEG are supported.'}), 400
        
        # Save and process image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            office_days, home_days = process_calendar(filepath)
            return jsonify({'office_days': office_days, 'home_days': home_days})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('index.html')

def process_calendar(image_path):
    """Detect icons in the calendar image."""
    office_lower = np.array([50, 50, 200])   # Lower bound for office icon color
    office_upper = np.array([150, 150, 255]) # Upper bound for office icon color

    home_lower = np.array([200, 50, 50])    # Lower bound for home icon color
    home_upper = np.array([255, 150, 150])  # Upper bound for home icon color

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to load image. Check file format and upload a valid image.")
    
    # Convert image to HSV (optional for better color matching)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect office and home icons using inRange
    office_mask = cv2.inRange(image, office_lower, office_upper)
    home_mask = cv2.inRange(image, home_lower, home_upper)

    # Count non-zero pixels in masks
    office_days = cv2.countNonZero(office_mask) // 1000  # Scale factor for approximation
    home_days = cv2.countNonZero(home_mask) // 1000

    return office_days, home_days

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT not provided
    app.run(host='0.0.0.0', port=port, debug=True)
