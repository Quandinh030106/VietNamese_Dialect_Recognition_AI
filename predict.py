import torch
import librosa
import argparse
from src.model import DialectClassifier
from src.utils import IDX_TO_CLS
from src.dataset import get_valid_transforms

def predict_file(audio_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = DialectClassifier(num_classes=6).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Chưa tìm thấy file model. Hãy train trước!")
        return

    # 2. Xử lý audio
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Không thể đọc file audio: {e}")
        return

    transforms = get_valid_transforms()
    audio_tensor = torch.from_numpy(audio).float()
    
    # Transform: [n_mels, time] -> unsqueeze -> [1, n_mels, time] -> transpose -> [1, time, n_mels] ??
    # Theo dataset.py: transforms trả về [n_mels, time] -> transpose -> [time, n_mels]
    # Collate fn: pad -> [B, T, M] -> unsqueeze -> transpose -> [B, 1, M, T]
    
    # Làm thủ công giống collate_fn cho 1 mẫu
    spec = transforms(audio_tensor).squeeze(0).transpose(0, 1) # [Time, Mel]
    input_length = torch.tensor([spec.shape[0]])
    
    # Thêm batch dimension và reshape chuẩn [Batch, 1, Mel, Time]
    spec = spec.unsqueeze(0) # [1, Time, Mel]
    spec = spec.unsqueeze(1).transpose(2, 3) # [1, 1, Mel, Time]
    
    spec = spec.to(device)
    input_length = input_length.to(device)

    # 3. Predict
    with torch.no_grad():
        output = model(spec, input_length)
        probs = torch.softmax(output, dim=1)
        score, predicted_idx = torch.max(probs, 1)
        
    idx = predicted_idx.item()
    region = IDX_TO_CLS[idx]
    confidence = score.item()
    
    print(f"--- KẾT QUẢ DỰ ĐOÁN ---")
    print(f"File: {audio_path}")
    print(f"Vùng miền: {region}")
    print(f"Độ tin cậy: {confidence*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán giọng vùng miền")
    parser.add_argument("--file", type=str, required=True, help="Đường dẫn file wav")
    parser.add_argument("--model", type=str, default="saved_models/best_dialect_classifier.pth", help="Đường dẫn file model")
    
    args = parser.parse_args()
    predict_file(args.file, args.model)