import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.detection.crash import CarCrashDetector
from app.attention.model import CNN_TSM_Attn_Enhanced
from app.attention.visualization import process_video_with_attention
from app.utils.config import UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR
from ultralytics import YOLO
import torch

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

@app.get("/")
def read_index():
    root_index = os.path.join(os.path.dirname(__file__), 'index.html')
    return FileResponse(root_index, media_type='text/html')

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    yolo_model = YOLO(os.path.join(MODEL_DIR, 'yolo11s.pt'))
    detector = CarCrashDetector(yolo_model)
    result = detector.run_and_save(input_path, os.path.join(OUTPUT_DIR, 'annotated_' + os.path.basename(input_path)))
    cnn_model = CNN_TSM_Attn_Enhanced()
    cnn_weights = os.path.join(MODEL_DIR, 'cnn_tsm_cbam_attn.pt')
    if os.path.exists(cnn_weights):
        state_dict = torch.load(cnn_weights, map_location='cpu')
        model_dict = cnn_model.state_dict()
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        cnn_model.load_state_dict(filtered_dict, strict=False)
    heatmap_out = os.path.join(OUTPUT_DIR, 'cnn_heatmap_' + os.path.basename(input_path))
    cnn_prediction, _, _ = process_video_with_attention(cnn_model, input_path, heatmap_out)
    return JSONResponse({
        "annotated_clip": '/static/' + 'annotated_' + os.path.basename(input_path),
        "cnn_heatmap_video": '/static/' + 'cnn_heatmap_' + os.path.basename(input_path),
        "cnn_prediction": float(cnn_prediction)
    }) 