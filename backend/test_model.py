# test_model.py
import os
from model_api import predict_image

# ✅ Path to your test image
TEST_IMAGE_PATH = "test_images/xray_image_2.png"  # <-- change to your real image path

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"❌ Image not found at {TEST_IMAGE_PATH}. Please check your path.")
else:
    print("🧠 Running DenseNet prediction test...")
    label, confidence, heatmap_url = predict_image(TEST_IMAGE_PATH)

    print("\n------ 🧩 MODEL TEST RESULT ------")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"GradCAM Path: {heatmap_url}")
    print("----------------------------------")

    # Optional: open Grad-CAM image if on desktop
    if heatmap_url and os.path.exists(heatmap_url.replace("/static", "static")):
        try:
            import webbrowser
            local_path = heatmap_url.replace("/static", "static")
            print(f"🌡 Opening GradCAM image: {local_path}")
            webbrowser.open(local_path)
        except Exception as e:
            print(f"⚠️ Could not open GradCAM: {e}")
