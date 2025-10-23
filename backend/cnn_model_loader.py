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
HEATMAP_FOLDER = os.path.join("static", "heatmaps")
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

# ------------------ LOAD MODEL ------------------
def load_densenet_model():
    """Load DenseNet121 architecture and pretrained weights."""
    try:
        print("🧠 Loading DenseNet121...")
        base_model = DenseNet121(weights=None, include_top=False, input_shape=(320, 320, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(len(LABELS), activation="sigmoid")(x)
        model = Model(inputs=base_model.input, outputs=output)
        model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
        print("✅ DenseNet model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


# ------------------ PREPROCESS ------------------
def preprocess_image(img_path):
    """Preprocess image for CNN input."""
    img = image.load_img(img_path, target_size=(320, 320))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# ------------------ GRADCAM ------------------
def generate_gradcam(img_path, model, last_conv_layer="conv5_block16_concat"):
    """Generate GradCAM overlay for model prediction."""
    img_array = preprocess_image(img_path)

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights.numpy())
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (320, 320))
    cam = cam / cam.max() if cam.max() != 0 else cam

    img = cv2.imread(img_path)
    img = cv2.resize(img, (320, 320))
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    filename = os.path.basename(img_path)
    output_path = os.path.join(HEATMAP_FOLDER, f"{os.path.splitext(filename)[0]}_gradcam.jpg")
    cv2.imwrite(output_path, superimposed)

    return output_path, predictions.numpy()


# ------------------ PREDICT ------------------
model = load_densenet_model()

if model is None:
    print("❌ Model failed to load at startup!")
else:
    print("✅ Model successfully initialized and ready for inference!")

def predict_xray(img_path):
    """Run CNN inference and GradCAM generation."""
    if model is None:
        raise RuntimeError("❌ Model not loaded!")

    gradcam_path, predictions = generate_gradcam(img_path, model)
    pred_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    label = LABELS[pred_index]
    return label, confidence, gradcam_path
