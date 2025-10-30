from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import requests
import uuid


# ------------------ LOAD ENVIRONMENT ------------------
from dotenv import load_dotenv
load_dotenv(override=True)

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

# ------------------ HUGGING FACE MODEL API ------------------
# (Replace with your actual Space URL)
HUGGINGFACE_MODEL_API = "https://noobmaster27-ddx.hf.space/"

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

# ------------------ FUNCTION: SEND IMAGE TO HUGGING FACE ------------------
def call_huggingface_model(image_path):
    try:
        print(f"📤 Sending image to Hugging Face model: {HUGGINGFACE_MODEL_API}")
        with open(image_path, "rb") as f:
            response = requests.post(HUGGINGFACE_MODEL_API, files={"data": f}, timeout=120)

        if response.status_code != 200:
            print("❌ Hugging Face error:", response.text)
            return None

        result = response.json()
        print("🧠 HF Response:", result)

        # Adjust this depending on your HF output structure
        if "data" in result:
            predictions = result["data"][0]
            top_label = list(predictions.keys())[0]
            top_conf = list(predictions.values())[0]
            gradcam_url = result["data"][1] if len(result["data"]) > 1 else None

            return {
                "label": top_label,
                "confidence": top_conf,
                "gradcam_url": gradcam_url
            }

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

        # Save uploaded image locally
        filename = f"{uuid.uuid4()}_{image_file.filename}"
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        image_file.save(image_path)
        print(f"📸 Saved uploaded image: {image_path}")

        # Call Hugging Face model API
        print("🧠 Sending to Hugging Face for prediction...")
        prediction = call_huggingface_model(image_path)

        if not prediction:
            return jsonify({"error": "Model inference failed"}), 500

        cnn_output = prediction["label"]
        confidence = prediction["confidence"]
        gradcam_url = prediction.get("gradcam_url")

        analysis_output = f"Detected: {cnn_output} (Confidence: {confidence*100:.2f}%)"

        # Save case to PostgreSQL
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
        print("✅ Case saved successfully to PostgreSQL!")

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
@app.route("/api/doctor/cases", methods=["GET"])
def get_all_cases():
    try:
        cases = PatientCase.query.all()
        print(f"📄 Retrieved {len(cases)} cases from DB.")
        return jsonify([case.to_dict() for case in cases]), 200
    except Exception as e:
        print(f"❌ Error fetching cases: {e}")
        return jsonify({"error": "Failed to fetch cases"}), 500


# ------------------ FRONTEND ROUTE ------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(os.path.dirname(__file__), "dist")
    if not os.path.exists(dist_dir):
        return jsonify({"message": "✅ Backend is running and connected to DB!"})

    if path != "" and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    else:
        return send_from_directory(dist_dir, "index.html")


# ------------------ MAIN ENTRY ------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database tables created successfully!")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
