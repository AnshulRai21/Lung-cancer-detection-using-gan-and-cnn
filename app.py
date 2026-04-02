from flask import Flask

app = Flask(__name__)


@app.route('/')
def home():
    return 'Lung CT Enhancement & Cancer Detection app is running.'


if __name__ == '__main__':
    app.run(debug=True)
