from flask import Flask, render_template, request
import os
from model_utils import predict_image
import uuid
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

# Make sure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure folders exist
OUTPUT_FOLDER = os.path.join(os.getcwd(), 'outputs')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER


@app.route('/')
def home():
    return render_template('index.html')  # Make sure index.html is in the templates folder

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('image')
        
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # Predict the result (ensure predict_image handles exceptions if any)
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

if __name__ == '__main__':
    app.run()
