# model_api.py
import os
from cnn_model_loader import predict_xray

def predict_image(img_path):
    """
    Wrapper for model prediction using cnn_model_loader.
    Returns:
        cnn_output (str): predicted disease label
        confidence (float): model confidence
        heatmap_info (dict): {web_path, local_path}
    """
    try:
        predicted_label, confidence, gradcam_path, prob_dict = predict_xray(img_path)

        if not gradcam_path:
            return predicted_label, confidence, None

        web_rel_path = f"/static/heatmaps/{os.path.basename(gradcam_path)}"
        abs_path = os.path.abspath(gradcam_path)

        heatmap_info = {
            "web_path": web_rel_path,
            "local_path": abs_path,
        }

        return predicted_label, confidence, heatmap_info

    except Exception as e:
        print("❌ Error in model_api.predict_image:", e)
        return "Error", 0.0, None
