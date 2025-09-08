# Satellite Image Classifier

This project classifies satellite images (AnnualCrop, Industrial, Pasture, Residential, SeaLake, Highway, River) using a TensorFlow Lite model served via a Flask backend, with a frontend (GeoScan) for interaction.

Live app: [https://satellite-image-classifier.vercel.app/](https://satellite-image-classifier.vercel.app/)

If you want to run this project locally on your laptop, you will need to make a few changes in both the backend (main.py) and the frontend (index.html).

---

## Requirements

* Python 3.9+
* Node.js + npm/yarn (optional, for serving the frontend locally)
* pip / virtualenv
* TensorFlow (CPU version is enough)
* Flask + Flask-CORS
* Pillow (PIL)
* NumPy

---

## Project Structure

```
.
├── backend/
│   ├── main.py
│   ├── models/
│   │   └── single_model_quantized.tflite
├── frontend/
│   └── index.html
└── README.md
└── requirements.txt

```

---

## Running Locally

1. Clone the repository

```
git clone https://github.com/your-username/satellite-image-classifier.git
cd satellite-image-classifier
pip install -r requirements.txt
```

2. Setup backend

```
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

requirements.txt should contain:

```
flask
flask-cors
tensorflow
pillow
numpy
```

3. Run backend

```
python main.py
```

The backend will be available at
[http://127.0.0.1:5000](http://127.0.0.1:5000)

4. Fix the frontend

Open frontend/index.html.
Find the line:

```
const backendUrl = "https://satellite-image-classifier.onrender.com";
```

Change it to:

```
const backendUrl = "http://127.0.0.1:5000";
```

5. Serve frontend

Option A: Open index.html directly in your browser.
Option B (recommended to avoid CORS/browser issues):

```
cd frontend
python -m http.server 3000
```

Now visit:
[http://127.0.0.1:3000](http://127.0.0.1:3000)

---

## Things You Must Change to Run Locally

1. Backend URL in index.html must point to [http://127.0.0.1:5000](http://127.0.0.1:5000)
2. Model file must exist at backend/models/single\_model\_quantized.tflite
3. Ports: Flask runs on 5000 by default, adjust if needed

---

## Troubleshooting

* ModuleNotFoundError: run pip install -r requirements.txt
* Model not found: check that models/single\_model\_quantized.tflite exists
* Frontend not updating results: make sure you changed backendUrl in index.html
* CORS issues: Flask already has CORS(app) enabled in main.py

---

## Deployments

* Frontend: Vercel (static hosting)
* Backend: Render (Flask + Python)

For local use, follow the steps above.

---
