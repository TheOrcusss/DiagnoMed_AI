from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
import os
from model_api import predict_image

# ------------------ ENV & APP CONFIG ------------------
load_dotenv(override=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------ DATABASE CONFIG ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ------------------ FILE CONFIG ------------------
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["HEATMAP_FOLDER"] = os.path.join("static", "heatmaps")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["HEATMAP_FOLDER"], exist_ok=True)


# ------------------ DATABASE MODEL ------------------
class PatientCase(db.Model):
    __tablename__ = "patient_case"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    blood_type = db.Column(db.String(10))
    symptoms = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    gradcam_url = db.Column(db.String(255))
    cnn_output = db.Column(db.String(255))
    analysis_output = db.Column(db.String(255))
    confidence = db.Column(db.Float)

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
            "analysis_output": self.analysis_output,
            "confidence": self.confidence,
        }


# ------------------ PATIENT SUBMISSION ROUTE ------------------
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

        # Save uploaded file directly (no secure_filename or UUID)
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)
        image_url = f"/static/uploads/{image_file.filename}"

        # Run CNN prediction
        prediction = predict_image(image_path)
        if prediction is None:
            return jsonify({"error": "Model inference failed"}), 500

        # Store CNN + GradCAM results
        cnn_output = prediction["label"]
        confidence = prediction["confidence"]
        gradcam_url = prediction["gradcam_web"]
        analysis_output = f"Detected condition: {cnn_output} (Confidence: {confidence*100:.2f}%)"

        # Save record to DB
        case = PatientCase(
            patient_name=name,
            age=age,
            blood_type=blood_type,
            symptoms=symptoms,
            image_url=image_url,
            gradcam_url=gradcam_url,
            cnn_output=cnn_output,
            confidence=confidence,
            analysis_output=analysis_output,
        )
        db.session.add(case)
        db.session.commit()

        return jsonify({
            "message": "Case saved successfully",
            "cnn_output": cnn_output,
            "confidence": confidence,
            "gradcam_url": gradcam_url,
            "image_url": image_url,
        }), 200

    except Exception as e:
        print(f"❌ Error in /api/patient/submit: {e}")
        return jsonify({"error": str(e)}), 500


# ------------------ DOCTOR VIEW CASES ROUTE ------------------
@app.route("/api/doctor/cases", methods=["GET"])
def get_all_cases():
    try:
        cases = PatientCase.query.all()
        return jsonify([case.to_dict() for case in cases])
    except Exception as e:
        print("❌ Error fetching cases:", e)
        return jsonify({"error": "Database fetch error"}), 500


# ------------------ MAIN ENTRY ------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("✅ Database ready and tables created.")
    app.run(host="0.0.0.0", port=5000)
