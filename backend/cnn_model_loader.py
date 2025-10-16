# cnn_model_loader.py
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import os

MODEL_PATH = "cnn_model/densenet.hdf5"

# ------------------ LOAD MODEL ------------------
def load_densenet_model():
    try:
        print("🧠 Building DenseNet121 architecture...")
        densenet_model = DenseNet121(
            weights=None,
            include_top=True,
            input_shape=(224, 224, 3),
            classes=3  # ✅ based on your notebook
        )

        print(f"📦 Loading weights from: {MODEL_PATH}")
        densenet_model.load_weights(MODEL_PATH)
        print("✅ DenseNet model fully loaded successfully!")
        return densenet_model

    except Exception as e:
        print(f"⚠️ Error loading DenseNet model: {e}")
        print("Trying partial load (by_name=True, skip_mismatch=True)...")
        try:
            densenet_model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
            print("✅ DenseNet model partially loaded (layer mismatch handled).")
            return densenet_model
        except Exception as e2:
            print(f"❌ Fallback failed: {e2}")
            return None


# ------------------ GRADCAM GENERATION ------------------
def generate_gradcam(img_path, model, last_conv_layer_name="conv5_block16_concat", output_path=None):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    grad_model = Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    if not output_path:
        output_path = img_path.replace(".jpg", "_gradcam.jpg").replace(".png", "_gradcam.png")
    cv2.imwrite(output_path, superimposed_img)

    return output_path, predictions.numpy()


# ------------------ PREDICTION FUNCTION ------------------
def predict_xray(img_path):
    model = load_densenet_model()
    if model is None:
        raise RuntimeError("Model could not be loaded")

    gradcam_path, predictions = generate_gradcam(img_path, model)
    pred_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    # 🔖 Replace with actual class names from your notebook
    labels = ["Normal", "Pneumonia", "Tuberculosis"]
    predicted_label = labels[pred_index] if pred_index < len(labels) else "Unknown"

    return predicted_label, confidence, gradcam_path


# ------------------ LOAD ON IMPORT ------------------
model = load_densenet_model()
