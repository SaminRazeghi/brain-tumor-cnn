# brain-tumor-cnn
Lightweight CNN-based brain tumor classifier using PyTorch, optimized for CPU deployment. Supports multi-class classification (glioma, meningioma, pituitary, no tumor) from MRI scans with robust preprocessing, data augmentation, training visualization, and evaluation.

## Features
- ✅ End-to-end training and evaluation pipeline
- 🧠 Multi-class classification (glioma, meningioma, pituitary, no tumor)
- 💻 CPU-only optimized (macOS compatible)
- 🛠️ Custom PyTorch `Dataset` with error handling
- 🔁 Real-time training logs and automatic model checkpointing
- 📊 Visualization: loss/accuracy curves, confusion matrix, ROC (for binary)
- 🧪 Outputs results in PNG and JSON formats
- 🧼 Robust image preprocessing and augmentation
- 📂 Modular code structure and clear logging for reproducibility

## Requirements
Install dependencies with:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn opencv-python pillow
