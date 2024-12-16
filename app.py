from flask import Flask, render_template, request, jsonify
import pytesseract
from PIL import Image
import re

app = Flask(__name__)

# Function to count icons
def count_icons(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    # Use regex to count occurrences of 🏠 and 🏢
    home_count = len(re.findall(r'🏠', text))
    office_count = len(re.findall(r'🏢', text))

    return {"home": home_count, "office": office_count}

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        file_path = f"./uploads/{file.filename}"
        file.save(file_path)

        # Process the uploaded image
        counts = count_icons(file_path)
        return jsonify(counts)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
