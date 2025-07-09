import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from .model import CNN_TSM_Attn_Enhanced

def create_attention_heatmap(attention_map, original_frame, alpha=0.6):
    att_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    h, w = original_frame.shape[:2]
    att_resized = cv2.resize(att_norm, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * att_resized), cv2.COLORMAP_JET)
    result = cv2.addWeighted(original_frame, 1-alpha, heatmap, alpha, 0)
    return result, att_resized

def visualize_channel_attention(channel_attention, frame_idx):
    weights = channel_attention.squeeze().cpu().numpy()
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(weights)), weights)
    plt.title(f'Channel Attention Weights - Frame {frame_idx}')
    plt.xlabel('Channel Index')
    plt.ylabel('Attention Weight')
    plt.tight_layout()
    return plt.gcf()

def process_video_with_attention(model, video_path, output_path, max_frames=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    from app.utils.video import preprocess_video, save_attention_video
    clip, raw_frames = preprocess_video(video_path, max_frames)
    clip = clip.to(device)
    attention_frames = []
    spatial_attention_maps = []
    channel_attention_data = []
    with torch.no_grad():
        pred = model(clip).item()
        for t in range(len(raw_frames)):
            frame_tensor = clip[:, :, t:t+1, :, :]
            backbone_features = model.backbone(frame_tensor.squeeze(2))
            cbam_output = model.cbam(backbone_features)
            spatial_att = model.cbam.spatial_attention.squeeze().cpu().numpy()
            channel_att = model.cbam.channel_attention.squeeze().cpu().numpy()
            original_frame = raw_frames[t]
            att_frame, att_map = create_attention_heatmap(spatial_att, original_frame)
            attention_frames.append(att_frame)
            spatial_attention_maps.append(att_map)
            channel_attention_data.append(channel_att)
    if attention_frames:
        save_attention_video(attention_frames, output_path)
    from app.attention.summary import generate_attention_summary
    generate_attention_summary(spatial_attention_maps, channel_attention_data, output_path.replace('.mp4', '_summary.png'))
    return pred, spatial_attention_maps, channel_attention_data 