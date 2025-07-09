import sys
from app.detection.crash import CarCrashDetector
from app.attention.model import CNN_TSM_Attn_Enhanced
from app.attention.visualization import process_video_with_attention
from app.utils.config import MODEL_DIR, OUTPUT_DIR
from app.utils.video import preprocess_video
from ultralytics import YOLO
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m app.main <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    yolo_model = YOLO(os.path.join(MODEL_DIR, 'yolo11s.pt'))
    detector = CarCrashDetector(yolo_model)
    detector.run_and_save(video_path, os.path.join(OUTPUT_DIR, 'annotated_' + os.path.basename(video_path)))
    cnn_model = CNN_TSM_Attn_Enhanced()
    cnn_weights = os.path.join(MODEL_DIR, 'cnn_tsm_cbam_attn.pt')
    if os.path.exists(cnn_weights):
        import torch
        state_dict = torch.load(cnn_weights, map_location='cpu')
        model_dict = cnn_model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        cnn_model.load_state_dict(filtered_dict, strict=False)
    heatmap_out = os.path.join(OUTPUT_DIR, 'cnn_heatmap_' + os.path.basename(video_path))
    process_video_with_attention(cnn_model, video_path, heatmap_out)

if __name__ == "__main__":
    main() 