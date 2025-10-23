from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from model_api import predict_image
import os, uuid

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------ DB CONFIG ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///health.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ------------------ DB MODEL ------------------
class PatientCase(db.Model):
    __tablename__ = "patient_case"
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    blood_type = db.Column(db.String(10))
    symptoms = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    gradcam_url = db.Column(db.String(255))
    cnn_output = db.Column(db.Text)
    analysis_output = db.Column(db.Text)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ------------------ ROUTES ------------------
@app.route("/api/patient/submit", methods=["POST"])
def upload_scan():
    """Upload image → preprocess → infer → store → return result."""
    try:
        name = request.form.get("name", "Anonymous")
        symptoms = request.form.get("symptoms", "")
        image = request.files.get("image")

        if not image:
            return jsonify({"error": "No image uploaded"}), 400

        # Save uploaded file
        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        
        filename = f"{uuid.uuid4()}_{image.filename}"
        img_path = os.path.join(upload_folder, filename)
        image.save(img_path)

        image_url = f"/static/uploads/{filename}"

        # Run CNN inference (preprocessing + GradCAM)
        result = predict_image(img_path)
        
        if not result:
            return jsonify({"error": "Model failed to process image"}), 500

        label = result["label"]
        confidence = result["confidence"]
        gradcam_url = result["gradcam_web"]

        # Save to database
        case = PatientCase(
            patient_name=name,
            symptoms=symptoms,
            image_url=image_url,
            gradcam_url=gradcam_url,
            cnn_output=f"{label} ({confidence*100:.2f}%)",
            analysis_output="Awaiting doctor’s review"
        )
        
        db.session.add(case)
        db.session.commit()

        return jsonify({
            "message": "Scan processed successfully",
            "cnn_output": case.cnn_output,
            "confidence": confidence,
            "image_url": image_url,
            "gradcam_url": gradcam_url
        })

    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({"error": "Failed to process image"}), 500


@app.route("/api/doctor/cases", methods=["GET"])
def get_cases():
    """Fetch all stored patient cases."""
    try:
        cases = PatientCase.query.all()
        return jsonify([c.to_dict() for c in cases])
    except Exception as e:
        print("❌ Fetch error:", e)
        return jsonify({"error": "Database fetch failed"}), 500


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("📦 Database initialized")
    app.run(host="0.0.0.0", port=5000, debug=True)
