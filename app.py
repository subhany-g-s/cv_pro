from flask import Flask, render_template, request, send_from_directory
import os
from model_utils import predict_image
import uuid
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/')
def home():
    return render_template('index.html')  # Ensure index.html is in the templates folder

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('image')
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            try:
                output_path, label = predict_image(upload_path)

                # Copy to outputs folder for serving
                output_filename = os.path.basename(output_path)
                served_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                shutil.copy(output_path, served_path)

                return render_template('upload.html',
                                       image_url=output_filename,
                                       label=label)
            except Exception as e:
                return render_template('upload.html', error="Error processing the image: " + str(e))
        else:
            return render_template('upload.html', error="Invalid file type. Please upload a valid image.")

    return render_template('upload.html')

# Serve the processed output image
@app.route('/outputs/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
