from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from dotenv import load_dotenv
import os, uuid
from model_api import predict_image  # 🧠 Model prediction + GradCAM generator

load_dotenv(override=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------ DATABASE CONFIG ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ------------------ DATABASE MODEL ------------------
class PatientCase(db.Model):
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    blood_type = db.Column(db.String(5))
    symptoms = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    gradcam_url = db.Column(db.String(255))
    gradcam_local = db.Column(db.String(255))
    cnn_output = db.Column(db.Text)
    analysis_output = db.Column(db.Text)

# ------------------ FILE CONFIG ------------------
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["HEATMAP_FOLDER"] = "static/heatmaps"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["HEATMAP_FOLDER"], exist_ok=True)

# ------------------ PATIENT SUBMIT ROUTE ------------------
@app.route("/api/patient/submit", methods=["POST"])
def patient_submit():
    try:
        name = request.form.get("name")
        age = request.form.get("age")
        blood_type = request.form.get("blood_type")
        symptoms = request.form.get("symptoms")
        file = request.files.get("image")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        base_url = request.host_url.rstrip("/")
        public_url = f"{base_url}/static/uploads/{filename}"

        # Run model prediction
        cnn_output, confidence, heatmap_info = predict_image(filepath)

        gradcam_web_url = (
            f"{base_url}{heatmap_info['web_path']}" if heatmap_info else None
        )
        gradcam_local_path = heatmap_info["local_path"] if heatmap_info else None
        analysis_output = f"Detected: {cnn_output} (Confidence: {confidence:.2f})"

        case = PatientCase(
            patient_name=name,
            age=age,
            blood_type=blood_type,
            symptoms=symptoms,
            image_url=public_url,
            gradcam_url=gradcam_web_url,
            gradcam_local=gradcam_local_path,
            cnn_output=cnn_output,
            analysis_output=analysis_output,
        )
        db.session.add(case)
        db.session.commit()

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

# ------------------ DOCTOR VIEW ------------------
@app.route("/api/doctor/cases", methods=["GET"])
def doctor_cases():
    try:
        cases = PatientCase.query.order_by(PatientCase.id.desc()).all()
        return jsonify([
            {
                "id": c.id,
                "patient_name": c.patient_name,
                "age": c.age,
                "blood_type": c.blood_type,
                "symptoms": c.symptoms,
                "image_url": c.image_url,
                "gradcam_url": c.gradcam_url,
                "cnn_output": c.cnn_output,
                "analysis_output": c.analysis_output,
            } for c in cases
        ]), 200
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

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000)
