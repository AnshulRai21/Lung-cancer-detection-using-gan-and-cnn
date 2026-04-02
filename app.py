import os
from datetime import datetime

import cv2
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.enhancement import enhance_image, load_gan_model
from utils.prediction import load_prediction_model, predict_cancer
from utils.preprocessing import preprocess_image
from utils.visualization import (
    apply_contrast_enhancement,
    generate_binarized_image,
    generate_zoomed_view,
)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load models once at startup
GAN_STATUS = load_gan_model(os.path.join('models', 'gan_generator.pth'))
PREDICTION_STATUS = load_prediction_model(os.path.join('models', 'cancer_classifier.pth'))


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def to_web_path(path: str) -> str:
    return '/' + path.replace('\\', '/')


@app.route('/', methods=['GET'])
def landing():
    return render_template(
        'index.html',
        gan_meta=GAN_STATUS,
        prediction_meta=PREDICTION_STATUS,
    )


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('image')

    if not file or not file.filename or not allowed_file(file.filename):
        return render_template(
            'index.html',
            gan_meta=GAN_STATUS,
            prediction_meta=PREDICTION_STATUS,
            error='Please upload a valid PNG/JPG/JPEG image.',
        )

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    base_name = secure_filename(file.filename).rsplit('.', 1)[0]
    stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')

    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{base_name}_{stamp}.png')
    file.save(upload_path)

    try:
        original = cv2.imread(upload_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            raise ValueError('Unable to read uploaded image.')

        preprocessed = preprocess_image(upload_path)
        enhanced = enhance_image(preprocessed)

        contrast = apply_contrast_enhancement(enhanced)
        zoomed = generate_zoomed_view(enhanced)
        binary = generate_binarized_image(enhanced)

        paths = {
            'original': os.path.join(app.config['OUTPUT_FOLDER'], f'original_{stamp}.png'),
            'enhanced': os.path.join(app.config['OUTPUT_FOLDER'], f'enhanced_{stamp}.png'),
            'contrast': os.path.join(app.config['OUTPUT_FOLDER'], f'contrast_{stamp}.png'),
            'zoomed': os.path.join(app.config['OUTPUT_FOLDER'], f'zoomed_{stamp}.png'),
            'binary': os.path.join(app.config['OUTPUT_FOLDER'], f'binary_{stamp}.png'),
        }

        cv2.imwrite(paths['original'], cv2.resize(original, (128, 128)))
        cv2.imwrite(paths['enhanced'], enhanced)
        cv2.imwrite(paths['contrast'], contrast)
        cv2.imwrite(paths['zoomed'], zoomed)
        cv2.imwrite(paths['binary'], binary)

        label, confidence = predict_cancer(enhanced)

        results = {
            'images': {key: to_web_path(value) for key, value in paths.items()},
            'label': label,
            'confidence': confidence,
            'gan_meta': GAN_STATUS,
            'prediction_meta': PREDICTION_STATUS,
        }

        return render_template('result.html', results=results)
    except ValueError as exc:
        return render_template(
            'index.html',
            gan_meta=GAN_STATUS,
            prediction_meta=PREDICTION_STATUS,
            error=str(exc),
        )


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
