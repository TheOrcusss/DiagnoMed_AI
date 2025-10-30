import requests
import os

# Hugging Face Space API URL – replace with your own Space
HF_SPACE_API = "https://NoobMaster27-DDX.hf.space/"

def predict_image(image_path):
    """
    Sends an image to your Hugging Face Space for prediction and Grad-CAM generation.
    Returns a dictionary with label, confidence, and Grad-CAM URL.
    """
    try:
        print(f"🧠 Sending image to Hugging Face API: {HF_SPACE_API}")

        with open(image_path, "rb") as f:
            response = requests.post(HF_SPACE_API, files={"img": f})

        if response.status_code != 200:
            print("❌ API call failed:", response.text)
            return None

        data = response.json()
        print("✅ API Response:", data)

        # Parse response from Hugging Face Space
        predictions = data.get("data", [{}])[0]
        gradcam_url = data.get("data", [None, None])[1]

        if not predictions:
            return None

        # Extract top label and confidence
        top_label = max(predictions.items(), key=lambda x: x[1])[0]
        confidence = predictions[top_label]

        result = {
            "label": top_label,
            "confidence": confidence,
            "all_predictions": predictions,
            "gradcam_web": f"https://NoobMaster27-DDX.hf.space{gradcam_url}" if gradcam_url else None
        }

        return result

    except Exception as e:
        print("⚠️ Error calling Hugging Face API:", e)
        return None
