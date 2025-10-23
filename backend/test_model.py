import os
import sys
import cv2
from cnn_model_loader import predict_xray, model, LABELS

def main():
    # 1️⃣  Pick image path (from arg or default folder)
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        uploads_dir = os.path.join("static", "uploads")
        if not os.path.exists(uploads_dir):
            print(f"❌ No uploads folder found: {uploads_dir}")
            return
        files = [f for f in os.listdir(uploads_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        if not files:
            print(f"⚠️ No image found in {uploads_dir}. Please upload one via the app first.")
            return
        img_path = os.path.join(uploads_dir, files[0])

    print(f"🩻 Testing model with image:\n   {img_path}")

    # 2️⃣  Run prediction
    try:
        predicted_label, confidence, gradcam_path, prob_dict = predict_xray(img_path)
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return

    # 3️⃣  Display results
    print("\n===== MODEL OUTPUT =====")
    print(f"Predicted label : {predicted_label}")
    print(f"Confidence      : {confidence * 100:.2f}%")
    print(f"GradCAM path    : {gradcam_path}")
    print("========================\n")

    # Show top 5 probabilities
    print("Top 5 probabilities:")
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    for label, prob in sorted_probs:
        print(f"  {label:20s} : {prob:.4f}")

    # 4️⃣  Optionally display GradCAM overlay
    if gradcam_path and os.path.exists(gradcam_path):
        print("\n🖼️ Opening GradCAM visualization window... (press any key to close)")
        img = cv2.imread(gradcam_path)
        if img is not None:
            cv2.imshow("GradCAM Overlay", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("⚠️ Unable to load GradCAM image for display.")
    else:
        print("⚠️ GradCAM image not found.")

if __name__ == "__main__":
    main()
