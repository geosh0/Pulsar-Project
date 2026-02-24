import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image
import os

# --- 1. CONFIGURATION ---
# ⚠️ FORCE CPU TO FIX "NO KERNEL IMAGE" ERROR
DEVICE = torch.device("cpu")
print(f"⚙️ forcing device: {DEVICE} (to bypass CUDA/GPU compatibility issues)")

# --- 2. DATASET CLASS ---
class PulsarDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, is_train=False, img_size=128):
        self.data = pd.read_csv(csv_file)
        self.is_train = is_train
        self.img_size = img_size
        
        self.resize_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        
        self.augment = transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prof_path = row['profile_path']
        base_dir = os.path.dirname(os.path.dirname(prof_path))
        fname = os.path.basename(prof_path)
        core_id = fname.replace("_Profile.png", "")
        
        features = ["Profile", "Time_Phase", "Freq_Phase", "DM_Curve"]
        images_dict = {}
        
        for feat in features:
            p = os.path.join(base_dir, feat, f"{core_id}_{feat}.png")
            try:
                img = Image.open(p)
                tensor = self.resize_tensor(img)
                
                if self.is_train:
                    tensor = self.augment(tensor)
                
                # Normalization
                mean, std = tensor.mean(), tensor.std()
                if std > 0: 
                    tensor = (tensor - mean) / std
                else: 
                    tensor = tensor - mean
                
                images_dict[feat.lower()] = tensor
            except:
                images_dict[feat.lower()] = torch.zeros((1, self.img_size, self.img_size))

        label = torch.tensor(row['label'], dtype=torch.float32)
        return images_dict, label

# --- 3. MODEL CLASS ---
class MultiStreamCNN(nn.Module):
    def __init__(self):
        super(MultiStreamCNN, self).__init__()
        
        def create_branch():
            return nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )

        self.branch_profile = create_branch()
        self.branch_time = create_branch()
        self.branch_freq = create_branch()
        self.branch_dm = create_branch()
        
        flat_size = 64 * 16 * 16
        total_features = flat_size * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out1 = self.branch_profile(x['profile'])
        out2 = self.branch_time(x['time_phase'])
        out3 = self.branch_freq(x['freq_phase'])
        out4 = self.branch_dm(x['dm_curve'])
        combined = torch.cat((out1, out2, out3, out4), dim=1)
        return self.classifier(combined)

# --- 4. TRAINING FUNCTION ---
def train_model(train_csv, val_csv, epochs=15, batch_size=32, lr=0.001):
    print(f"🚀 Training starting on {DEVICE}...")

    # Load Data
    train_data = PulsarDataset(train_csv, is_train=True)
    val_data   = PulsarDataset(val_csv, is_train=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Init Model
    model = MultiStreamCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # TRAIN
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        # Wrapped in try-except to catch interruptions safely
        try:
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                labels = labels.to(DEVICE).unsqueeze(1)
                imgs_gpu = {k: v.to(DEVICE) for k, v in images.items()}
                
                optimizer.zero_grad()
                outputs = model(imgs_gpu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        except Exception as e:
            print(f"Error during training step: {e}")
            break
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.to(DEVICE).unsqueeze(1)
                imgs_gpu = {k: v.to(DEVICE) for k, v in images.items()}
                
                outputs = model(imgs_gpu)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = val_correct / val_total
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
        print(f"    Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        print(f"    Val   Loss: {val_epoch_loss:.4f} | Acc: {val_epoch_acc:.4f}")

    _plot_history(history)
    return model

# --- 5. PLOTTING HELPER ---
def _plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', color='orange')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 6. EVALUATION FUNCTION ---
def evaluate_model(model, test_csv, batch_size=32):
    print(f"\n--- Final Evaluation on Test Set ---")
    test_data = PulsarDataset(test_csv, is_train=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    model.eval()
    model.to(DEVICE) # Ensure model is on the right device
    
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Predicting"):
            imgs_gpu = {k: v.to(DEVICE) for k, v in images.items()}
            
            outputs = model(imgs_gpu)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred).flatten()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n🏆 FINAL TEST RESULTS 🏆")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Negatives: {cm[0][0]} | False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]} | True Positives:  {cm[1][1]}")