import os
from cnn_model_loader import predict_xray

def predict_image(img_path):
    """Run model inference and return prediction + heatmap."""
    try:
        label, confidence, gradcam_path = predict_xray(img_path)

        if not gradcam_path or not os.path.exists(gradcam_path):
            raise FileNotFoundError("GradCAM file not found")

        return {
            "label": label,
            "confidence": confidence,
            "gradcam_web": f"/static/heatmaps/{os.path.basename(gradcam_path)}",
            "gradcam_local": os.path.abspath(gradcam_path)
        }
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        return None
