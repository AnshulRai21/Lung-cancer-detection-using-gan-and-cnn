# Lung CT Image Enhancement and Cancer Detection (Step 1)

This is the initial setup for a Flask-based capstone project.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── templates/
├── static/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   └── outputs/
├── utils/
├── models/
└── uploads/
```

## Setup Instructions

1. Create and activate a virtual environment:
   - Linux/macOS:
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open in browser:
   - http://127.0.0.1:5000/

---

> Disclaimer: This system assists medical professionals and does not replace diagnosis.
