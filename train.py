import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import process_filepath, CLS_TO_IDX
from src.dataset import AudioDataset, collate_fn, get_train_transforms, get_valid_transforms
from src.model import DialectClassifier

# --- CONFIG ---
CSV_PATH = "data/VNspeech.csv"
DATA_FOLDER = "data/dataset/dataset" # Đường dẫn đến folder chứa 01, 02...
MODEL_SAVE_PATH = "saved_models/best_dialect_classifier.pth"
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load và xử lý dữ liệu
    if not os.path.exists(CSV_PATH):
        print(f"Không tìm thấy file {CSV_PATH}. Hãy kiểm tra đường dẫn.")
        return

    df = pd.read_csv(CSV_PATH)
    df = process_filepath(df, DATA_FOLDER)
    
    # Map nhãn
    df["class_name"] = df["class_name"].map(CLS_TO_IDX)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["filename"], df["class_name"], test_size=0.2, random_state=42, stratify=df["class_name"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Convert to list
    X_train, y_train = X_train.tolist(), y_train.tolist()
    X_val, y_val = X_val.tolist(), y_val.tolist()

    # Datasets & DataLoaders
    train_ds = AudioDataset(X_train, y_train, transforms=get_train_transforms())
    val_ds = AudioDataset(X_val, y_val, transforms=get_valid_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # 2. Khởi tạo Model
    model = DialectClassifier(num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 3. Training Loop
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")
        for specs, labels, lens in train_pbar:
            specs, labels, lens = specs.to(device), labels.to(device), lens.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs, lens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * specs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            train_pbar.set_postfix(loss=train_loss/total_train, acc=train_correct/total_train)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for specs, labels, lens in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]"):
                specs, labels, lens = specs.to(device), labels.to(device), lens.to(device)
                outputs = model(specs, lens)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * specs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val

        print(f"Summary E{epoch}: Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

        # Save Best Model
        if epoch_val_loss < best_val_loss:
            print(f"Validation Loss improved. Saving model to {MODEL_SAVE_PATH}...")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()