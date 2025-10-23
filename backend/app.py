from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from dotenv import load_dotenv
import os, uuid
from model_api import predict_image  # 🧠 Model prediction + GradCAM generator

# ------------------ ENV & APP CONFIG ------------------
load_dotenv(override=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------ DATABASE CONFIG ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ------------------ FILE CONFIG ------------------
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["HEATMAP_FOLDER"] = os.path.join("static", "heatmaps")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["HEATMAP_FOLDER"], exist_ok=True)

# ------------------ DATABASE MODEL ------------------
class PatientCase(db.Model):
    __tablename__ = "patient_cases"

    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    blood_type = db.Column(db.String(5))
    symptoms = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    gradcam_url = db.Column(db.String(255))
    gradcam_local = db.Column(db.String(255))
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

# ------------------ PATIENT SUBMIT ROUTE ------------------
@app.route("/api/patient/submit", methods=["POST"])
def patient_submit():
    try:
        # Get patient data
        name = request.form.get("name")
        age = request.form.get("age")
        blood_type = request.form.get("blood_type")
        symptoms = request.form.get("symptoms")
        file = request.files.get("image")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # Save uploaded file
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        base_url = request.host_url.rstrip("/")
        public_url = f"{base_url}/static/uploads/{filename}"

        # ------------------ MODEL PREDICTION ------------------
        cnn_output, confidence, heatmap_info = predict_image(filepath)
        if not heatmap_info:
            raise RuntimeError("GradCAM generation failed")

        gradcam_web_url = f"{base_url}{heatmap_info['web_path']}"
        gradcam_local_path = heatmap_info["local_path"]
        analysis_output = f"Detected: {cnn_output} (Confidence: {confidence:.2f})"

        # ------------------ SAVE TO DATABASE ------------------
        case = PatientCase(
            patient_name=name,
            age=int(age) if age else None,
            blood_type=blood_type,
            symptoms=symptoms,
            image_url=public_url,
            gradcam_url=gradcam_web_url,
            gradcam_local=gradcam_local_path,
            cnn_output=cnn_output,
            confidence=confidence,
            analysis_output=analysis_output,
        )
        db.session.add(case)
        db.session.commit()

        print(f"✅ Case saved: {case.id} ({cnn_output} - {confidence:.2f})")

        return jsonify({
            "message": "Case submitted successfully!",
            "cnn_output": cnn_output,
            "confidence": confidence,
            "analysis_output": analysis_output,
            "image_url": public_url,
            "gradcam_url": gradcam_web_url,
        }), 200

    except Exception as e:
        print("❌ Error during submission:", e)
        return jsonify({"error": str(e)}), 500


# ------------------ DOCTOR VIEW (All Cases) ------------------
@app.route("/api/doctor/cases", methods=["GET"])
def doctor_cases():
    try:
        cases = PatientCase.query.order_by(PatientCase.id.desc()).all()
        return jsonify([c.to_dict() for c in cases]), 200
    except Exception as e:
        print("❌ Fetch error:", e)
        return jsonify({"error": str(e)}), 500


# ------------------ FRONTEND SERVE ------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(os.path.dirname(__file__), "dist")
    if not os.path.exists(dist_dir):
        return jsonify({"message": "✅ Backend connected and running!"})
    if path != "" and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    else:
        return send_from_directory(dist_dir, "index.html")


# ------------------ MAIN ENTRY ------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("📦 Database tables ensured.")
    app.run(host="0.0.0.0", port=5000, debug=True)
