from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import os
import uuid
from gradio_client import Client, handle_file

# ------------------ LOAD ENVIRONMENT ------------------
load_dotenv(override=True)
print("Database URL:", os.getenv("DATABASE_URL"))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ------------------ FLASK CONFIG ------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------ DATABASE CONFIG ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ------------------ FILE PATHS ------------------
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ------------------ HUGGING FACE MODEL CONFIG ------------------
HF_SPACE = "NoobMaster27/DDX"  # Your Hugging Face Space name
client = Client(HF_SPACE)
print(f"✅ Connected to Hugging Face Space: {HF_SPACE}")

# ------------------ DATABASE MODEL ------------------
class PatientCase(db.Model):
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    blood_type = db.Column(db.String(10))
    symptoms = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    gradcam_url = db.Column(db.String(255))
    cnn_output = db.Column(db.Text)
    confidence = db.Column(db.Float)
    analysis_output = db.Column(db.Text)

    def to_dict(self):
        return {
            "id": self.id,
            "patient_name": self.patient_name,
            "age": self.age,
            "blood_type": self.blood_type,
            "symptoms": self.symptoms,
            "image_url": self.image_url,
            "gradcam_url": self.gradcam_url,
            "cnn_output": self.cnn_output,
            "confidence": self.confidence,
            "analysis_output": self.analysis_output,
        }

# ------------------ CALL HUGGING FACE MODEL ------------------

def call_huggingface_model(image_path):
    try:
        print(f"📤 Sending image to Hugging Face model: {HF_SPACE}")

        response = requests.post(
            f"{HF_SPACE}/predict",
            files={"img": open(image_path, "rb")},
            timeout=120,
        )

        print(f"🧠 Status Code: {response.status_code}")
        if response.status_code != 200:
            print("❌ Hugging Face API Error:", response.text)
            return None

        result = response.json()
        print("🧩 Full HF Response:", result)

        # Extract predictions and gradcam from result
        if "data" in result and isinstance(result["data"], list):
            data = result["data"][0]
            if isinstance(data, dict) and "predictions" in data:
                preds = data["predictions"]
                top_label = max(preds, key=preds.get)
                confidence = preds[top_label]
                gradcam_url = data.get("gradcam_url")
                return {
                    "label": top_label,
                    "confidence": confidence,
                    "gradcam_url": gradcam_url,
                }

        print("⚠️ Unexpected response format received.")
        return None

    except Exception as e:
        print(f"❌ Exception calling Hugging Face: {e}")
        return None


# ------------------ PATIENT UPLOAD ROUTE ------------------
@app.route("/api/patient/submit", methods=["POST"])
def submit_patient_case():
    try:
        name = request.form.get("name", "Anonymous")
        age = request.form.get("age")
        blood_type = request.form.get("blood_type")
        symptoms = request.form.get("symptoms", "")
        image_file = request.files.get("image")

        if not image_file:
            return jsonify({"error": "No image uploaded"}), 400

        # Save uploaded image
        filename = f"{uuid.uuid4()}_{image_file.filename}"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(image_path)
        print(f"📸 Saved image at {image_path}")

        # Run model inference
        prediction = call_huggingface_model(image_path)
        if not prediction:
            return jsonify({"error": "Model inference failed"}), 500

        predictions = prediction.get("predictions", {})
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_text = ", ".join([f"{k} ({v*100:.2f}%)" for k, v in sorted_preds])

        # Best prediction
        cnn_output = sorted_preds[0][0]
        confidence = sorted_preds[0][1]
        gradcam_url = prediction.get("gradcam_url")

        analysis_output = f"Top 3 Predictions: {top3_text}"

        # Save to PostgreSQL
        case = PatientCase(
            patient_name=name,
            age=age,
            blood_type=blood_type,
            symptoms=symptoms,
            image_url=f"/static/uploads/{filename}",
            gradcam_url=gradcam_url,
            cnn_output=cnn_output,
            confidence=confidence,
            analysis_output=analysis_output,
        )

        db.session.add(case)
        db.session.commit()
        print("✅ Case saved to database!")

        return jsonify({
            "message": "Case submitted successfully!",
            "cnn_output": cnn_output,
            "confidence": confidence,
            "gradcam_url": gradcam_url,
            "image_url": f"/static/uploads/{filename}"
        }), 200

    except Exception as e:
        print(f"❌ Error in /api/patient/submit: {e}")
        return jsonify({"error": str(e)}), 500

# ------------------ DOCTOR FETCH ALL CASES ------------------
@app.route('/api/doctor/cases', methods=['GET'])
def get_doctor_cases():
    try:
        cases = PatientCase.query.all()
        return jsonify([
            {
                "id": c.id,
                "patient_name": c.patient_name,
                "age": c.age,
                "blood_type": c.blood_type,
                "symptoms": c.symptoms,
                "cnn_output": c.cnn_output,
                "analysis_output": c.analysis_output,
                "image_url": c.image_url,
                "gradcam_url": c.gradcam_url,
            }
            for c in cases
        ])
    except Exception as e:
        print("❌ Error fetching doctor cases:", e)
        return jsonify({"error": "Server error"}), 500


# ------------------ FRONTEND ROUTE ------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(os.path.dirname(__file__), "dist")
    if not os.path.exists(dist_dir):
        return jsonify({"message": "✅ Backend running, DB connected!"})

    if path != "" and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    else:
        return send_from_directory(dist_dir, "index.html")

# ------------------ MAIN ENTRY ------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database initialized successfully.")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
