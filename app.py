import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.enhancement import enhance_image, load_gan_model
from utils.prediction import load_prediction_model, predict_cancer
from utils.preprocessing import preprocess_image
from utils.visualization import (
    apply_contrast_enhancement,
    generate_binarized_image,
    generate_zoomed_view,
    save_visual_outputs,
)

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

GAN_STATUS = load_gan_model(os.path.join('models', 'gan_generator.pth'))
PREDICTION_STATUS = load_prediction_model(os.path.join('models', 'cancer_classifier.pth'))


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            try:
                processed = preprocess_image(save_path)
                enhanced = enhance_image(processed)

                original = processed.squeeze()
                original_u8 = (original * 255.0).clip(0, 255).astype('uint8')
                contrast = apply_contrast_enhancement(enhanced)
                zoomed = generate_zoomed_view(enhanced)
                binary = generate_binarized_image(enhanced)

                output_paths = save_visual_outputs(
                    original=original_u8,
                    enhanced=enhanced,
                    contrast=contrast,
                    zoomed=zoomed,
                    binary=binary,
                    output_dir=app.config['OUTPUT_FOLDER'],
                )

                web_paths = {
                    key: f"/{path.replace(os.sep, '/')}"
                    for key, path in output_paths.items()
                }
                # Use the normalized original CT view for classification to stay
                # aligned with typical classifier training inputs.
                label, confidence = predict_cancer(original_u8)

                return render_template(
                    'result.html',
                    results={
                        'label': label,
                        'confidence': confidence,
                        'images': web_paths,
                        'gan_meta': GAN_STATUS,
                        'prediction_meta': PREDICTION_STATUS,
                    },
                )
            except ValueError as exc:
                return render_template('index.html', error=str(exc), gan_meta=GAN_STATUS)

    return render_template('index.html', gan_meta=GAN_STATUS)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
