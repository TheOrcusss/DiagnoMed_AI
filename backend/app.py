from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os, uuid, numpy as np, cv2
from dotenv import load_dotenv
from tensorflow.keras.preprocessing import image
from cnn_model_loader import model as densenet_model  # ✅ preloaded DenseNet
import tensorflow.keras.backend as K
from flask_migrate import Migrate

load_dotenv(override=True)

# ------------------ APP SETUP ------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ------------------ DATABASE CONFIG ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# ------------------ MODEL ------------------
class PatientCase(db.Model):
    id = db.Column(db.String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_name = db.Column(db.String(120))
    age = db.Column(db.Integer)
    blood_type = db.Column(db.String(5))
    symptoms = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    gradcam_url = db.Column(db.String(255))
    cnn_output = db.Column(db.Text)
    analysis_output = db.Column(db.Text)

# ------------------ FILE UPLOAD CONFIG ------------------
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["HEATMAP_FOLDER"] = "static/heatmaps"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["HEATMAP_FOLDER"], exist_ok=True)


# ------------------ GRADCAM FUNCTION ------------------
def generate_gradcam(model, img_array, original_path, last_conv_layer_name="conv5_block16_2_conv"):
    """Generates and saves Grad-CAM heatmap for a given image."""
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

        # 🩻 Overlay Grad-CAM on the original image
        img = cv2.imread(original_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        heatmap_filename = f"{uuid.uuid4()}_gradcam.jpg"
        heatmap_path = os.path.join(app.config["HEATMAP_FOLDER"], heatmap_filename)
        cv2.imwrite(heatmap_path, superimposed_img)

        return f"{request.host_url.rstrip('/')}/static/heatmaps/{heatmap_filename}"

    except Exception as e:
        print("⚠️ GradCAM generation failed:", e)
        return None


# ------------------ PREDICTION FUNCTION ------------------
def predict_image(img_path):
    """Runs DenseNet prediction and returns CNN + GradCAM outputs."""
    if densenet_model is None:
        return "Model not loaded", "⚠️ DenseNet model unavailable", None

    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = densenet_model.predict(img_array)
        class_idx = np.argmax(preds[0])

        # ✅ Match your .ipynb labels
        labels = ["NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
        cnn_output = labels[class_idx]
        confidence = preds[0][class_idx]
        analysis_output = f"Detected: {cnn_output} (Confidence: {confidence:.2f})"

        # ✅ Generate GradCAM
        heatmap_url = generate_gradcam(densenet_model, img_array, img_path)

        return cnn_output, analysis_output, heatmap_url

    except Exception as e:
        print("❌ Prediction error:", e)
        return "Error", str(e), None


# ------------------ PATIENT SUBMISSION ROUTE ------------------
@app.route("/api/patient/submit", methods=["POST"])
def patient_submit():
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
    public_url = f"{base_url}/static/uploads/{filename}"

    # ✅ Model prediction + GradCAM
    cnn_output, analysis_output, heatmap_url = predict_image(filepath)

    # ✅ Save in DB
    case = PatientCase(
        patient_name=name,
        age=age,
        blood_type=blood_type,
        symptoms=symptoms,
        image_url=public_url,
        gradcam_url=heatmap_url,
        cnn_output=cnn_output,
        analysis_output=analysis_output,
    )
    db.session.add(case)
    db.session.commit()

    return jsonify({
        "message": "Case submitted successfully!",
        "cnn_output": cnn_output,
        "analysis_output": analysis_output,
        "image_url": public_url,
        "heatmap_url": heatmap_url
    }), 200


# ------------------ DOCTOR FETCH CASES ------------------
@app.route("/api/doctor/cases", methods=["GET"])
def doctor_cases():
    cases = PatientCase.query.all()
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
    app.run(host="0.0.0.0", port=5000)
