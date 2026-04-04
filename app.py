import os
import cv2   # ✅ ADD THIS
import torch
from flask import Flask, render_template
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.enhancement import enhance_image, load_gan_model
from utils.preprocessing import preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

GAN_STATUS = load_gan_model(os.path.join('models', 'gan_generator.pth'))


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    image_url = None
    enhanced_url = None
    preprocess_meta = None
    gan_meta = GAN_STATUS
    error = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            image_url = save_path.replace('\\', '/')

            try:
                processed = preprocess_image(save_path)
                preprocess_meta = {
                    'shape': processed.shape,
                    'min': float(processed.min()),
                    'max': float(processed.max()),
                }

                enhanced = enhance_image(processed)
                enhanced_name = f"enhanced_{filename.rsplit('.', 1)[0]}.png"
                enhanced_path = os.path.join(app.config['OUTPUT_FOLDER'], enhanced_name)
                cv2.imwrite(enhanced_path, enhanced)
                enhanced_url = enhanced_path.replace('\\', '/')
            except ValueError as exc:
                error = str(exc)

    return render_template(
        'index.html',
        image_url=image_url,
        enhanced_url=enhanced_url,
        preprocess_meta=preprocess_meta,
        gan_meta=gan_meta,
        error=error,
    )


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
