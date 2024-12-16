from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import numpy as np
import os
import datetime

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

def get_current_date_position(image):
    """Find the position of the current date circled in blue."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the bounding box for the largest blue contour (assumed to be the date circle)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return y + h  # Return y-coordinate of the bottom of the circle
    return None

def process_calendar(image_path):
    """Detect icons in the calendar image using edge detection and color filtering."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to load image. Check file format and upload a valid image.")

    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for gray and blue
    gray_lower = np.array([0, 0, 50])
    gray_upper = np.array([180, 50, 200])
    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])

    # Create masks for gray and blue colors
    mask_gray = cv2.inRange(hsv, gray_lower, gray_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    combined_mask = cv2.bitwise_or(mask_gray, mask_blue)

    # Get the position of the current date
    current_date_y = get_current_date_position(image)
    if current_date_y is None:
        raise ValueError("Current date not found in the image.")

    # Perform edge detection on the combined mask
    edges = cv2.Canny(combined_mask, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize counts
    office_icon_count = 0
    home_icon_count = 0

    # Analyze contours
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Only count icons that are above the current date row
        if y <= current_date_y and 50 < area < 10000:
            roi = combined_mask[y:y+h, x:x+w]
            non_zero_ratio = cv2.countNonZero(roi) / (w * h)
            if non_zero_ratio > 0.5:
                if w > h:  # Horizontal icons -> Office
                    office_icon_count += 1
                else:      # Vertical icons -> Home
                    home_icon_count += 1

    return office_icon_count, home_icon_count

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT not provided
    app.run(host='0.0.0.0', port=port, debug=True)
