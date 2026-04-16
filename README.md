import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)

model.load_state_dict(torch.load("/kaggle/working/best_model.pth", map_location=device))
model = model.to(device)
# model.eval()








import pandas as pd
import torch
import cv2
import numpy as np
from torchvision import transforms
from collections import deque

test_csv_path = "/kaggle/input/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance/data/test.csv"
base_path = "/kaggle/input/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance/data"  
num_frames = 6
img_size = 96
threshold = 0.3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= TRANSFORM 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames-1, num_frames).astype(int)
    
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    return frames

def classify(prob):
    if prob < 0.3:
        return "Normal"
    elif prob < 0.6:
        return "Suspicious"
    else:
        return "High Anomaly "

def confidence_level(prob):
    if prob < 0.2 or prob > 0.8:
        return "High"
    elif prob < 0.4 or prob > 0.6:
        return "Medium"
    else:
        return "Low"

# ========= LOAD TEST CSV =========
df = pd.read_csv(test_csv_path)

# fix path if needed
if "video_name" in df.columns:
    df["video_name"] = df["video_name"].apply(lambda x: base_path + "/" + x)
else:
    raise ValueError("Column 'video_path' not found in test.csv")

model.eval()

for idx, row in df.iterrows():
    video_path = row["video_name"]
    
    frames = extract_frames(video_path)
    if frames is None:
        print(f"{video_path} → Skipped")
        continue
    
    frames = [transform(f) for f in frames]
    frames = torch.stack(frames)
    frames = frames.permute(1, 0, 2, 3)
    frames = frames.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(frames).squeeze()
        prob = torch.sigmoid(output).item()
    
    score = prob * 10
    label = classify(prob)
    confidence = confidence_level(prob)
    
    print(f"{video_path.split('/')[-1]} → Prob: {prob:.3f} | Score: {score:.2f} | Class: {label} | Confidence: {confidence}")
