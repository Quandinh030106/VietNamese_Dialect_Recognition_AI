import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil

class DialectClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DialectClassifier, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.lstm = nn.LSTM(input_size=128 * 20, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x, input_lengths):
        x = self.conv_blocks(x)
        new_lengths = input_lengths // 8
        new_lengths = torch.clamp(new_lengths, min=1)
        max_len = x.size(3)
        new_lengths = torch.clamp(new_lengths, max=max_len)
        x = x.transpose(1, 3).contiguous()
        B, T, H, C = x.shape
        x = x.view(B, T, H * C)
        x_packed = nn.utils.rnn.pack_padded_sequence(x, new_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x_packed, _ = self.lstm(x_packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        return self.fc_out(x)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DialectClassifier(num_classes=6).to(device)


try:
    model.load_state_dict(torch.load("best_dialect_classifier.pth", map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Mapping nhãn
IDX_TO_CLS = { 
    0: 'Giọng Bắc (North)', 
    1: 'Bắc Trung Bộ (North Central Coast)',
    2: 'Nam Trung Bộ (South Central Coast)', 
    3: 'Tây Nguyên (Central Highland)',
    4: 'Đông Nam Bộ (South East)',
    5: 'Tây Nam Bộ (South West)',
}

# Transform
valid_audio_transforms = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
    torchaudio.transforms.AmplitudeToDB()
)


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):

    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        audio, sr = librosa.load(temp_filename, sr=16000, mono=True)
        audio_tensor = torch.from_numpy(audio).float()
        
        # Chuyển đổi audio thành spectrogram
        spec = valid_audio_transforms(audio_tensor).squeeze(0).transpose(0, 1)
        
        input_length = torch.tensor([spec.shape[0]]).to(device)
        
        spec = spec.unsqueeze(0)
        spec = spec.unsqueeze(1).transpose(2, 3)
        spec = spec.to(device)
        
        # Dự đoán
        with torch.no_grad():
            output = model(spec, input_length)
            probs = torch.softmax(output, dim=1)
            score, predicted_idx = torch.max(probs, 1)
            
        result = {
            "prediction": IDX_TO_CLS[predicted_idx.item()],
            "confidence": float(score.item()),
            "probabilities": {IDX_TO_CLS[i]: float(probs[0][i]) for i in range(6)}
        }
        
        return result

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Xóa file tạm
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

