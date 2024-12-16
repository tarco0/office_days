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
    """Detect icons in the calendar image using color masking and edge detection."""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Unable to load image. Check file format and upload a valid image.")
    
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for gray tones (icons appear grayish)
    lower_gray = np.array([0, 0, 50])     # Lower bound of gray
    upper_gray = np.array([180, 50, 200]) # Upper bound of gray

    # Create a mask for gray colors
    mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Apply edge detection on the masked image
    edges = cv2.Canny(mask, threshold1=30, threshold2=100)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debug: Draw contours on a copy of the original image
    debug_image = image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("debug_contours.jpg", debug_image)  # Save for inspection

    # Initialize counts
    office_icon_count = 0
    home_icon_count = 0

    # Analyze contours
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Adjust area filtering threshold
        if 50 < area < 10000:  # Loosen lower limit, increase upper limit
            roi = mask[y:y+h, x:x+w]
            non_zero_ratio = cv2.countNonZero(roi) / (w * h)

            # Check if region is gray (high ratio of mask pixels)
            if non_zero_ratio > 0.5:
                if w > h:  # Horizontal icons -> Office
                    office_icon_count += 1
                else:      # Vertical icons -> Home
                    home_icon_count += 1

    # Debug counts
    print(f"Office Icons Detected: {office_icon_count}, Home Icons Detected: {home_icon_count}")

    # Return counts
    return office_icon_count, home_icon_count



if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT not provided
    app.run(host='0.0.0.0', port=port, debug=True)
