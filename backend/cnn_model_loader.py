import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

# ------------------ CONFIG ------------------
MODEL_PATH = os.path.join("cnn_model", "densenet.hdf5")

# Your actual class labels
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# ------------------ SAFE GPU/CPU SETUP ------------------
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow using GPU ({len(gpus)} available)")
    else:
        print("⚙️ Running on CPU (no GPU detected)")
except Exception as e:
    print("⚠️ GPU setup issue:", e)

# ------------------ LOAD MODEL ------------------
def load_densenet_model():
    """Loads DenseNet121 architecture and your trained weights."""
    try:
        print("🧠 Building DenseNet121 architecture...")
        model = DenseNet121(
            weights=None,
            include_top=True,
            input_shape=(224, 224, 3),
            classes=len(LABELS)
        )

        print(f"📦 Loading weights from: {MODEL_PATH}")
        model.load_weights(MODEL_PATH)
        print("✅ DenseNet model loaded successfully!")
        return model

    except Exception as e:
        print(f"⚠️ Full load failed: {e}")
        print("🔄 Trying partial layer load...")
        try:
            model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
            print("✅ DenseNet partially loaded (skip_mismatch=True)")
            return model
        except Exception as e2:
            print(f"❌ Could not load DenseNet weights: {e2}")
            return None


# ------------------ GRADCAM FUNCTION ------------------
def generate_gradcam(img_path, model, last_conv_layer_name="conv5_block16_concat", output_path=None):
    """Generates GradCAM overlay heatmap for a given image."""
    try:
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
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
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

    except Exception as e:
        print("⚠️ GradCAM generation failed:", e)
        return None, None


# ------------------ PREDICTION FUNCTION ------------------
def predict_xray(img_path):
    """Runs DenseNet prediction and GradCAM for chest X-ray."""
    if model is None:
        raise RuntimeError("❌ DenseNet model not loaded")

    gradcam_path, predictions = generate_gradcam(img_path, model)
    if predictions is None:
        raise RuntimeError("GradCAM generation failed")

    pred_index = np.argmax(predictions)
    confidence = float(np.max(predictions))
    predicted_label = LABELS[pred_index] if pred_index < len(LABELS) else "Unknown"

    # Create a dictionary for all class probabilities
    prob_dict = {LABELS[i]: float(predictions[0][i]) for i in range(len(LABELS))}

    return predicted_label, confidence, gradcam_path, prob_dict


# ------------------ LOAD ON IMPORT ------------------
model = load_densenet_model()
