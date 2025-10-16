# model_api.py
import os
import uuid
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from cnn_model_loader import model as densenet_model  # ✅ preloaded model from cnn_model_loader.py

# ------------------ LABELS (from your dataset) ------------------
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "PatientId", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# ------------------ GRADCAM GENERATION ------------------
def generate_gradcam(model, img_array, original_path, save_dir="static/heatmaps",
                     last_conv_layer_name="conv5_block16_concat"):
    """
    Generates Grad-CAM heatmap and saves it in static/heatmaps.
    Returns dict with both web and local paths.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)

        # Create Grad-CAM model
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

        # Overlay GradCAM on original image
        img = cv2.imread(original_path)
        if img is None:
            raise ValueError(f"Could not read image from path: {original_path}")

        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        # Save GradCAM
        heatmap_filename = f"{uuid.uuid4()}_gradcam.jpg"
        heatmap_path = os.path.join(save_dir, heatmap_filename)
        cv2.imwrite(heatmap_path, superimposed_img)

        abs_path = os.path.abspath(heatmap_path)
        print(f"✅ GradCAM saved: {abs_path}")

        return {
            "web_path": f"/static/heatmaps/{heatmap_filename}",
            "local_path": abs_path
        }

    except Exception as e:
        print("⚠️ GradCAM generation failed:", e)
        return None


# ------------------ PREDICTION FUNCTION ------------------
def predict_image(img_path):
    """
    Run DenseNet model on image and return results.
    Returns:
        cnn_output (str): predicted label
        confidence (float): confidence score
        heatmap_info (dict): contains web_path and local_path for GradCAM
    """
    if densenet_model is None:
        print("❌ Model not loaded!")
        return "Model not loaded", 0.0, None

    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        preds = densenet_model.predict(img_array)
        class_idx = np.argmax(preds[0])
        confidence = float(preds[0][class_idx])

        cnn_output = LABELS[class_idx] if class_idx < len(LABELS) else "Unknown"

        # Generate Grad-CAM
        heatmap_info = generate_gradcam(densenet_model, img_array, img_path)

        print(f"🧠 Prediction: {cnn_output} ({confidence*100:.2f}%)")
        if heatmap_info:
            print(f"🔥 GradCAM Path: {heatmap_info['local_path']}")

        return cnn_output, confidence, heatmap_info

    except Exception as e:
        print("❌ Prediction error:", e)
        return "Error", 0.0, None
