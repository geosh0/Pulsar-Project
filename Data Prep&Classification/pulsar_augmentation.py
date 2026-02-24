import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

# --- 1. FUNCTION TO GET TRANSFORMS ---
def get_transforms(img_size=128):
    """
    Returns (train_transform, test_transform)
    """
    # --- TRAINING TRANSFORM (The "Messy" Version) ---
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))
    ])

    # --- TESTING TRANSFORM (The "Clean" Version) ---
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    return train_transform, test_transform

# --- 2. THE DATASET CLASS ---
# (Must be a class for PyTorch, but we keep it simple)
class PulsarDataset(Dataset):
    def __init__(self, csv_file, transform, img_size=128):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prof_path = row['profile_path']
        
        # Logic: Go up two folders, then down into the specific feature folder
        base_dir = os.path.dirname(os.path.dirname(prof_path))
        fname = os.path.basename(prof_path)
        core_id = fname.replace("_Profile.png", "") 
        
        images_dict = {}
        features = ["Profile", "Time_Phase", "Freq_Phase", "DM_Curve"]
        
        for feat in features:
            image_path = os.path.join(base_dir, feat, f"{core_id}_{feat}.png")
            
            try:
                img = Image.open(image_path)
                tensor_img = self.transform(img)
                
                # Custom Normalization
                mean = tensor_img.mean()
                std = tensor_img.std()
                if std > 0:
                    tensor_img = (tensor_img - mean) / std
                else:
                    tensor_img = tensor_img - mean
                
                images_dict[feat] = tensor_img
                
            except Exception:
                # Return blank if missing
                images_dict[feat] = torch.zeros((1, self.img_size, self.img_size))

        label = torch.tensor(row['label'], dtype=torch.float32)
        return images_dict, label

# --- 3. FUNCTION TO VISUALIZE DATA ---
def show_example(csv_path, transform, index=0):
    """
    Loads the dataset and plots the 4 features for a specific index.
    """
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV not found at {csv_path}")
        return

    # Create the dataset temporarily just to view it
    dataset = PulsarDataset(csv_path, transform=transform)
    
    # Grab the item
    data_dict, label = dataset[index]
    print(f"✅ Loading Index {index} | Label: {label.item()}")
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    order = ["Profile", "Time_Phase", "Freq_Phase", "DM_Curve"]
    
    for i, name in enumerate(order):
        img = data_dict[name].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(name)
        axes[i].axis('off')
        
    plt.suptitle(f"Data Example (Index {index})", fontsize=14)
    plt.show()