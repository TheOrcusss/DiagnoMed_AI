from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_migrate import Migrate
from dotenv import load_dotenv
import os
import uuid

# ✅ Import model prediction logic
from model_api import predict_image

# ------------------ ENV + APP SETUP ------------------
load_dotenv(override=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------ DATABASE CONFIG ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
if app.config["SQLALCHEMY_DATABASE_URI"].startswith("postgres://"):
    app.config["SQLALCHEMY_DATABASE_URI"] = app.config["SQLALCHEMY_DATABASE_URI"].replace("postgres://", "postgresql://")

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ------------------ MODEL: DATABASE TABLE ------------------
class PatientCase(db.Model):
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    blood_type = db.Column(db.String(5))
    symptoms = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    gradcam_url = db.Column(db.String(255))        # 🌐 Public GradCAM URL
    gradcam_local = db.Column(db.String(255))      # 💾 Local file path
    cnn_output = db.Column(db.String(120))
    confidence = db.Column(db.Float)
    analysis_output = db.Column(db.Text)

# ------------------ FILE UPLOAD CONFIG ------------------
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["HEATMAP_FOLDER"] = "static/heatmaps"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["HEATMAP_FOLDER"], exist_ok=True)

# ------------------ PATIENT SUBMISSION ROUTE ------------------
@app.route("/api/patient/submit", methods=["POST"])
def patient_submit():
    try:
        # Get form data
        name = request.form.get("name")
        age = request.form.get("age")
        blood_type = request.form.get("blood_type")
        symptoms = request.form.get("symptoms")
        file = request.files.get("image")

        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        # ✅ Save uploaded file
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        base_url = request.host_url.rstrip("/")
        public_image_url = f"{base_url}/static/uploads/{filename}"

        # ✅ Run model prediction
        cnn_output, confidence, heatmap_info = predict_image(filepath)

        # ✅ Handle GradCAM URLs
        gradcam_web_url = None
        gradcam_local_path = None
        if heatmap_info:
            gradcam_web_url = f"{base_url}{heatmap_info['web_path']}"
            gradcam_local_path = heatmap_info["local_path"]

        # ✅ Analysis output
        analysis_output = f"Detected: {cnn_output} (Confidence: {confidence:.2f})"

        # ✅ Save case in DB
        new_case = PatientCase(
            patient_name=name,
            age=int(age) if age else None,
            blood_type=blood_type,
            symptoms=symptoms,
            image_url=public_image_url,
            gradcam_url=gradcam_web_url,
            gradcam_local=gradcam_local_path,
            cnn_output=cnn_output,
            confidence=confidence,
            analysis_output=analysis_output,
        )
        db.session.add(new_case)
        db.session.commit()

        print(f"✅ Saved case for {name}: {cnn_output} ({confidence:.2f})")
        print(f"🩻 GradCAM Path: {gradcam_local_path}")

        return jsonify({
            "message": "Case submitted successfully!",
            "cnn_output": cnn_output,
            "confidence": confidence,
            "analysis_output": analysis_output,
            "image_url": public_image_url,
            "gradcam_web_url": gradcam_web_url,
            "gradcam_local_path": gradcam_local_path
        }), 200

    except Exception as e:
        print("❌ Error during submission:", e)
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# ------------------ DOCTOR FETCH CASES ------------------
@app.route("/api/doctor/cases", methods=["GET"])
def doctor_cases():
    try:
        cases = PatientCase.query.order_by(PatientCase.id.desc()).all()
        data = []
        for c in cases:
            data.append({
                "id": c.id,
                "patient_name": c.patient_name,
                "age": c.age,
                "blood_type": c.blood_type,
                "symptoms": c.symptoms,
                "image_url": c.image_url,
                "gradcam_url": c.gradcam_url,
                "cnn_output": c.cnn_output,
                "confidence": c.confidence,
                "analysis_output": c.analysis_output,
            })
        return jsonify(data), 200
    except Exception as e:
        print("❌ Fetch error:", e)
        return jsonify({"error": str(e)}), 500

# ------------------ FRONTEND SERVE ------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(os.path.dirname(__file__), "dist")
    if not os.path.exists(dist_dir):
        return jsonify({"message": "✅ Backend running and connected to Render PostgreSQL!"})
    if path != "" and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    else:
        return send_from_directory(dist_dir, "index.html")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
