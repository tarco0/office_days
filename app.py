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
    """Detect icons in the calendar image using edge detection and color filtering."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to load image. Check file format and upload a valid image.")

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for gray, blue, and any other relevant icons
    gray_lower = np.array([0, 0, 50])      # Lower bound for gray
    gray_upper = np.array([180, 50, 200])  # Upper bound for gray

    blue_lower = np.array([90, 50, 50])    # Lower bound for blue tones
    blue_upper = np.array([130, 255, 255]) # Upper bound for blue tones

    # Create masks for gray and blue colors
    mask_gray = cv2.inRange(hsv, gray_lower, gray_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Combine masks
    combined_mask = cv2.bitwise_or(mask_gray, mask_blue)

    # Perform edge detection on the combined mask
    edges = cv2.Canny(combined_mask, threshold1=30, threshold2=100)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize counts
    office_icon_count = 0
    home_icon_count = 0

    # Analyze contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if 50 < area < 10000:  # Adjust area thresholds to detect relevant icons
            x, y, w, h = cv2.boundingRect(contour)
            roi = combined_mask[y:y+h, x:x+w]
            
            # Check if region is significant
            non_zero_ratio = cv2.countNonZero(roi) / (w * h)
            if non_zero_ratio > 0.5:
                if w > h:  # Horizontal icons -> Office
                    office_icon_count += 1
                else:      # Vertical icons -> Home
                    home_icon_count += 1

    # Return the counts
    return office_icon_count, home_icon_count

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT not provided
    app.run(host='0.0.0.0', port=port, debug=True)
