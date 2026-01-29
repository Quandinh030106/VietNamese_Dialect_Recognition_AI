import torch
import torch.nn as nn
import torchaudio
import librosa
from torch.utils.data import Dataset

# Định nghĩa Transforms
def get_train_transforms():
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
        torchaudio.transforms.AmplitudeToDB(),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35, p=0.6),
        torchaudio.transforms.TimeMasking(time_mask_param=35, p=0.6)
    )

def get_valid_transforms():
    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),
        torchaudio.transforms.AmplitudeToDB()
    )

class AudioDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = X  # List các đường dẫn file
        self.y = y  # List các nhãn (số)
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        path_audio = self.X[idx] 
        label = self.y[idx]
        
        # Load audio bằng librosa
        try:
            audio, sr = librosa.load(path_audio, sr=16000, mono=True)
            audio_tensor = torch.from_numpy(audio).float()

            if self.transforms:
                # Transform và đổi chiều [channels, n_mels, time] -> [n_mels, time] -> transpose
                spec = self.transforms(audio_tensor).squeeze(0).transpose(0, 1)
            return spec, label
        except Exception as e:
            print(f"Error loading {path_audio}: {e}")
            # Trả về tensor rỗng hoặc xử lý lỗi tùy ý
            return torch.zeros(1, 80), label

def collate_fn(batch):
    spectrograms = []
    labels = []
    input_lengths = []
    
    for (spec, label) in batch:
        if spec.size(0) > 0: # Kiểm tra nếu load lỗi
            spectrograms.append(spec) 
            labels.append(torch.tensor(label, dtype=torch.long)) 
            input_lengths.append(spec.shape[0])
        
    padded_spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True, padding_value=0.0)
    # Thêm chiều channel và transpose để khớp input Conv2d: [Batch, Channel, Height, Width]
    final_spectrograms = padded_spectrograms.unsqueeze(1).transpose(2, 3) 
    labels_tensor = torch.stack(labels)
    
    return final_spectrograms, labels_tensor, torch.tensor(input_lengths)