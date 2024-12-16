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
    """Detect icons in the calendar image using edge detection and gray color analysis."""
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to load image. Check file format and upload a valid image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Threshold the image to isolate gray tones
    _, binary_gray = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize counts
    office_icon_count = 0
    home_icon_count = 0

    # Analyze contours for approximate icon sizes and shapes
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:  # Filter contours by size (adjust as needed)
            x, y, w, h = cv2.boundingRect(contour)
            roi = binary_gray[y:y+h, x:x+w]
            
            # Check for predominantly gray regions
            if cv2.countNonZero(roi) > 0.5 * (w * h):
                if w > h:  # Example heuristic for distinguishing icons
                    office_icon_count += 1
                else:
                    home_icon_count += 1

    # Approximate days by scaling counts
    office_days = office_icon_count // 1  # Adjust scaling factor as needed
    home_days = home_icon_count // 1

    return office_days, home_days

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT not provided
    app.run(host='0.0.0.0', port=port, debug=True)
