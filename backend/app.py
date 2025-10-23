from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import os
from model_api import predict_image

# ------------------ LOAD ENVIRONMENT ------------------
load_dotenv(override=True)

# ------------------ FLASK CONFIG ------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------ DATABASE CONFIG ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ------------------ FILE PATHS ------------------
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["HEATMAP_FOLDER"] = "static/heatmaps"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["HEATMAP_FOLDER"], exist_ok=True)

# ------------------ DATABASE MODEL ------------------
class PatientCase(db.Model):
    id = db.Column(db.String, primary_key=True)
    patient_name = db.Column(db.String(120))
    symptoms = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    gradcam_url = db.Column(db.String(255))
    cnn_output = db.Column(db.Text)
    analysis_output = db.Column(db.Text)

    def to_dict(self):
        return {
            "id": self.id,
            "patient_name": self.patient_name,
            "symptoms": self.symptoms,
            "image_url": self.image_url,
            "gradcam_url": self.gradcam_url,
            "cnn_output": self.cnn_output,
            "analysis_output": self.analysis_output,
        }

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

        # Save uploaded file
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)
        print(f"📸 Image saved at: {image_path}")

        # Run prediction
        print("🧠 Running model inference...")
        prediction = predict_image(image_path)
        print(f"✅ Prediction result: {prediction}")

        if prediction is None:
            return jsonify({"error": "Model inference failed"}), 500

        cnn_output = prediction["label"]
        confidence = prediction["confidence"]
        gradcam_url = prediction["gradcam_web"]
        analysis_output = f"Detected condition: {cnn_output} (Confidence: {confidence*100:.2f}%)"

        # Save to DB
        case = PatientCase(
            patient_name=name,
            age=age,
            blood_type=blood_type,
            symptoms=symptoms,
            image_url=f"/static/uploads/{image_file.filename}",
            gradcam_url=gradcam_url,
            cnn_output=cnn_output,
            confidence=confidence,
            analysis_output=analysis_output,
        )
        db.session.add(case)
        db.session.commit()
        print("🧾 Case saved successfully to DB!")

        return jsonify({
            "message": "Case saved successfully",
            "cnn_output": cnn_output,
            "confidence": confidence,
            "gradcam_url": gradcam_url,
            "image_url": f"/static/uploads/{image_file.filename}"
        }), 200

    except Exception as e:
        print(f"❌ Error in /api/patient/submit: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------ DOCTOR FETCH ALL CASES ------------------
@app.route("/api/doctor/cases", methods=["GET"])
def get_all_cases():
    try:
        cases = PatientCase.query.all()
        print(f"🧾 Retrieved {len(cases)} cases from DB.")
        return jsonify([case.to_dict() for case in cases]), 200
    except Exception as e:
        print(f"❌ Error fetching cases: {e}")
        return jsonify({"error": "Failed to fetch cases"}), 500


# ------------------ SERVE FRONTEND (REACT / VITE) ------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(os.path.dirname(__file__), "dist")

    # If frontend isn't built yet
    if not os.path.exists(dist_dir):
        return jsonify({"message": "✅ Backend is running and connected to DB!"})

    # Serve static files or index.html
    if path != "" and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)
    else:
        return send_from_directory(dist_dir, "index.html")


# ------------------ MAIN ENTRY ------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database and tables initialized successfully!")
    app.run(host="0.0.0.0", port=5000)
