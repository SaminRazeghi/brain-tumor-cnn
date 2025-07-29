#!/usr/bin/env python3
"""
EE8601 Final Project: Automated Tumor Detection in Brain MRI Images
Author: [Your Name]
Date: July 29, 2025

This project implements a robust pipeline for tumor detection (e.g., meningioma) in brain MRI images
using PyTorch with a CNN-based classification approach, optimized for CPU on macOS.
Loads real dataset from provided archive paths.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from pathlib import Path
import cv2
import random
import json
import time
import os

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from pathlib import Path
    import cv2
    import random
    import json
    import time
    import os
except ImportError as e:
    logger.error(f"Missing required module {e}. Please install with 'pip install torch torchvision numpy matplotlib seaborn scikit-learn opencv-python pillow'.")
    print(f"Error: Missing required module {e}. Please install with 'pip install torch torchvision numpy matplotlib seaborn scikit-learn opencv-python pillow'.")
    exit(1)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Use CPU device (macOS does not support CUDA natively)
device = torch.device("cpu")
logger.info(f"Using device: {device}")
print(f"Using device: {device}")

class BrainTumorDataset(Dataset):
    """Custom dataset for brain tumor MRI images with shape validation"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(labels)))}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            if image.shape[0] != 3:
                raise ValueError(f"Expected 3 channels, got {image.shape[0]} for {self.image_paths[idx]}")
            return image, torch.tensor(self.class_to_idx[label], dtype=torch.long)
        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            return torch.zeros((3, 224, 224)), torch.tensor(0, dtype=torch.long)

class TumorCNN(nn.Module):
    """CNN model for tumor classification (multi-class)"""
    def __init__(self, num_classes):
        super(TumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class TumorDetectionPipeline:
    """Pipeline for tumor detection in brain MRI images"""
    def __init__(self, data_dir="/Users/saminrazeghi/Documents/Samin/TMU/Semester3-Summer/DS/Project/archive", model_dir="models", results_dir="results"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        for d in [self.model_dir, self.results_dir]:
            d.mkdir(exist_ok=True)
        self.device = device
        self.model = None
        self.metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.class_names = []  # Will be populated dynamically
    
    def load_real_dataset(self):
        """Load real brain tumor MRI dataset from provided paths"""
        logger.info("Loading real brain tumor MRI dataset from archive...")
        print("Loading real brain tumor MRI dataset from archive...")
        image_paths, labels = [], []
        
        # Load training and testing data
        for split in ["Training", "Testing"]:
            split_dir = self.data_dir / split
            if split_dir.exists():
                for cls in os.listdir(split_dir):
                    class_dir = split_dir / cls
                    if class_dir.is_dir():
                        self.class_names.append(cls)
                        for img_path in class_dir.glob("*.jpg"):  # Adjusted for .jpg extension
                            image_paths.append(str(img_path))
                            labels.append(cls)
        
        # Ensure unique class names
        self.class_names = sorted(list(set(self.class_names)))
        num_classes = len(self.class_names)
        logger.info(f"Detected classes: {self.class_names}")
        print(f"Detected classes: {self.class_names}")
        
        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)  # 60% train, 20% val, 20% test
        
        logger.info(f"Loaded {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples")
        print(f"Loaded {len(X_train)} training, {len(X_val)} validation, and {len(X_test)} test samples")
        return X_train, X_val, X_test, y_train, y_val, y_test, num_classes
    
    def setup_transforms(self):
        """Define image transforms"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return train_transform, val_test_transform
    
    def train_model(self, X_train, X_val, y_train, y_val, num_classes, epochs=25, batch_size=16):
        """Train the model"""
        logger.info("Starting model training...")
        print("Starting model training...")
        train_transform, val_transform = self.setup_transforms()
        train_dataset = BrainTumorDataset(X_train, y_train, train_transform)
        val_dataset = BrainTumorDataset(X_val, y_val, val_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        
        self.model = TumorCNN(num_classes).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_acc = 0.0, 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (torch.max(outputs, 1)[1] == labels).sum().item() / len(labels)
            
            val_loss, val_acc = self.validate(val_loader, criterion)
            self.metrics["train_loss"].append(train_loss / len(train_loader))
            self.metrics["train_acc"].append(train_acc / len(train_loader))
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_acc"].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss/len(train_loader):.4f}, "
                       f"Train Acc {train_acc/len(train_loader):.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
            print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss/len(train_loader):.4f}, "
                  f"Train Acc {train_acc/len(train_loader):.2f}%, Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), self.model_dir / "best_model.pth")
                logger.info(f"New best model saved with accuracy {best_acc:.2f}%")
                print("New best model saved!")
        
        return train_loader, val_loader
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                val_loss += criterion(outputs, labels).item()
                val_acc += (torch.max(outputs, 1)[1] == labels).sum().item() / len(labels)
        return val_loss / len(val_loader), 100 * val_acc / len(val_loader)
    
    def evaluate_and_visualize(self, val_loader, test_loader=None):
        """Evaluate and visualize results"""
        logger.info("Evaluating and visualizing model results...")
        print("Evaluating and visualizing model results...")
        self.model.load_state_dict(torch.load(self.model_dir / "best_model.pth"))
        self.model.eval()
        
        all_preds, all_labels, all_probs = [], [], []
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc = self.validate(val_loader, criterion)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                preds = torch.max(outputs, 1)[1]
                probs = torch.softmax(outputs, dim=1)[:, torch.argmax(probs, dim=1)] if len(self.class_names) > 2 else torch.softmax(outputs, dim=1)[:, 1]
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        auc = roc_auc_score(all_labels, all_probs) if len(self.class_names) == 2 else None
        logger.info(f"Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f if auc else 'N/A for multi-class'}")
        print(f"Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f if auc else 'N/A for multi-class'}")
        
        # Enhanced Visualizations
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics["train_loss"], label="Training Loss", color='blue')
        plt.plot(self.metrics["val_loss"], label="Validation Loss", color='red')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 2, 2)
        plt.plot(self.metrics["train_acc"], label="Training Accuracy", color='blue')
        plt.plot(self.metrics["val_acc"], label="Validation Accuracy", color='red')
        plt.title("Training and Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 2, 3)
        cm = confusion_matrix(all_labels, all_preds, labels=range(len(self.class_names)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        if auc is not None:  # Only plot ROC for binary classification
            plt.subplot(2, 2, 4)
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", color='darkorange')
            plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            plt.title("Receiver Operating Characteristic (ROC) Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(self.results_dir / "results.png", dpi=300)
        plt.close()
        
        results = {"accuracy": accuracy, "auc": auc, "metrics": self.metrics}
        with open(self.results_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {self.results_dir}")
        print(f"Results saved to {self.results_dir}")
        return results

def main():
    """Main execution function"""
    logger.info("=== Tumor Detection Pipeline Started ===")
    print("=== Tumor Detection Pipeline ===")
    pipeline = TumorDetectionPipeline()
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = pipeline.load_real_dataset()
    train_transform, val_test_transform = pipeline.setup_transforms()
    train_dataset = BrainTumorDataset(X_train, y_train, train_transform)
    val_dataset = BrainTumorDataset(X_val, y_val, val_test_transform)
    test_dataset = BrainTumorDataset(X_test, y_test, val_test_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=True)
    pipeline.train_model(X_train, X_val, y_train, y_val, num_classes)
    pipeline.evaluate_and_visualize(val_loader, test_loader)
    logger.info("=== Tumor Detection Pipeline Completed ===")
    print("=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()