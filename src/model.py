import torch
import torch.nn as nn

class DialectClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DialectClassifier, self).__init__()
        
        # 1. Convolutional Blocks
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        
        # 2. LSTM
        # Input size tính toán dựa trên output của Conv blocks
        rnn_input_size = 128 * 20 
        rnn_hidden_size = 128
        
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(rnn_hidden_size * 2, 64) # *2 vì Bidirectional
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x, input_lengths):
        # x shape: [Batch, 1, n_mels, time]
        x = self.conv_blocks(x) 
        
        # Tính toán lại chiều dài chuỗi sau khi qua các lớp Pooling
        # MaxPool (2,2) -> (2,2) -> (1,2) => Time dimension giảm: /2 /2 /2 = /8
        new_lengths = input_lengths // 8
        new_lengths = torch.clamp(new_lengths, min=1)
        
        # Đảm bảo length không vượt quá kích thước thực tế sau conv
        max_len = x.size(3)
        new_lengths = torch.clamp(new_lengths, max=max_len)

        # Reshape để đưa vào LSTM: [Batch, Time, Features]
        x = x.transpose(1, 3).contiguous() 
        B, T, H, C = x.shape
        x = x.view(B, T, H * C) 
        
        # Pack sequence để LSTM bỏ qua phần padding
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, new_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        x_packed, _ = self.lstm(x_packed)
        
        # Unpack
        x, _ = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
        
        # Global Average Pooling theo chiều thời gian
        x = torch.mean(x, dim=1) 
        
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x) 
        
        return x