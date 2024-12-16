from flask import Flask, request, render_template, jsonify
from PIL import Image
import cv2
import numpy as np
import os

app = Flask(__name__, static_url_path='/static')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Save and process image
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process image
        office_days, home_days = process_calendar(filepath)
        return jsonify({'office_days': office_days, 'home_days': home_days})
    
    return render_template('index.html')

def process_calendar(image_path):
    """Detect icons in the calendar image."""
    office_icon_color = [100, 100, 250]  # Approx color for office icon (example)
    home_icon_color = [250, 100, 100]    # Approx color for home icon (example)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return 0, 0

    office_days = 0
    home_days = 0

    # Scan image for colors resembling icons
    for row in image:
        for pixel in row:
            if np.allclose(pixel, office_icon_color, atol=50):
                office_days += 1
            elif np.allclose(pixel, home_icon_color, atol=50):
                home_days += 1

    # Return counts (divided by an arbitrary factor for approximate days)
    return office_days // 1000, home_days // 1000

if __name__ == '__main__':
    app.run(debug=True)
