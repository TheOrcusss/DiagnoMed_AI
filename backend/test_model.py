import os
import webbrowser
from model_api import predict_image

# ✅ Path to your test image
TEST_IMAGE_PATH = "test_images/xray_image.png"  # <-- change this if needed

if not os.path.exists(TEST_IMAGE_PATH):
    print(f"❌ Image not found at {TEST_IMAGE_PATH}. Please check your path.")
else:
    print("🧠 Running DenseNet prediction test...")
    label, confidence, heatmap_info = predict_image(TEST_IMAGE_PATH)

    print("\n------ 🧩 MODEL TEST RESULT ------")
    print(f"Predicted Label: {label}")
    print(f"Confidence: {confidence:.2f}")

    if isinstance(heatmap_info, dict):
        gradcam_local = heatmap_info.get("local_path")
        gradcam_web = heatmap_info.get("web_path")

        print(f"GradCAM Local Path: {gradcam_local}")
        print(f"GradCAM Web Path: {gradcam_web}")

        # ✅ Try opening the GradCAM in the default image viewer or browser
        if gradcam_local and os.path.exists(gradcam_local):
            try:
                print(f"🌡 Opening GradCAM image: {gradcam_local}")
                webbrowser.open(f"file:///{os.path.abspath(gradcam_local)}")
            except Exception as e:
                print(f"⚠️ Could not open GradCAM image: {e}")
        else:
            print("⚠️ GradCAM image file not found on disk.")
    else:
        print("⚠️ GradCAM path not returned by model.")
