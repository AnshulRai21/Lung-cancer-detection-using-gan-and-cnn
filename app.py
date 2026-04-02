import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from utils.preprocessing import preprocess_image

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def home():
    image_url = None
    preprocess_meta = None
    error = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
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
            except ValueError as exc:
                error = str(exc)

    return render_template(
        'index.html',
        image_url=image_url,
        preprocess_meta=preprocess_meta,
        error=error,
    )


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
