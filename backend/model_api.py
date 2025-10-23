import os
from cnn_model_loader import predict_xray

def predict_image(img_path):
    """Run CNN model inference and return prediction + GradCAM heatmap info."""
    try:
        # Run inference using cnn_model_loader
        label, confidence, heatmap_info = predict_xray(img_path)

        if not heatmap_info or not os.path.exists(heatmap_info["local_path"]):
            print("⚠️ GradCAM not found or generation failed.")
            return {
                "label": label,
                "confidence": confidence,
                "gradcam_web": None,
                "gradcam_local": None
            }

        print(f"✅ GradCAM generated: {heatmap_info['local_path']}")
        return {
            "label": label,
            "confidence": confidence,
            "gradcam_web": heatmap_info["web_path"],
            "gradcam_local": heatmap_info["local_path"],
        }

    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return {
            "label": "Error",
            "confidence": 0.0,
            "gradcam_web": None,
            "gradcam_local": None,
        }
