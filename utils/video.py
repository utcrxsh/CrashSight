import cv2
import numpy as np
torch = None
try:
    import torch
except ImportError:
    pass

def preprocess_video(video_path, max_frames=None, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    arr = np.stack(frames)  # [T, H, W, 3]
    arr = arr.transpose(3, 0, 1, 2)  # [3, T, H, W]
    arr = arr.astype(np.float32) / 255.0
    arr = torch.from_numpy(arr)
    arr = arr.unsqueeze(0)  # [1, 3, T, H, W]
    return arr, frames

def save_attention_video(frames, output_path, fps=8):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release() 