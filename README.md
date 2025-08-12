# Underwater Object Detection Project

This project implements underwater object detection using YOLOv8 to detect plastics and other objects in underwater environments. The model has been trained on a custom dataset and achieves good performance for underwater object detection tasks.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributors](#contributors)

## 🎯 Project Overview

This project focuses on detecting underwater objects, particularly plastics, using computer vision techniques. The implementation uses YOLOv8 (You Only Look Once version 8) with a custom-trained model specifically optimized for underwater environments. **This project was developed in Google Colab** and includes integrated data handling and visualization features.

### Key Features:
- YOLOv8-based object detection
- Optimized for underwater environments
- Custom training on plastics dataset
- Comprehensive evaluation metrics
- Easy-to-use inference pipeline
- Google Colab integration with data upload/download
- Interactive visualization and monitoring

## 🚀 Installation

### Option 1: Google Colab (Recommended)
This project was developed and tested in Google Colab. For the best experience:

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload the notebook**: Upload `UWOD.ipynb` to your Colab environment
3. **Install dependencies**: Run the installation cells in the notebook
4. **Upload your data**: Use the data upload cells provided in the notebook

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

#### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "Underwater Object Detection Group - 4"
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Verify Installation
```python
from ultralytics import YOLO
print("Installation successful!")
```

### Why Google Colab?
- **Free GPU access**: T4 or V100 GPUs available for faster training
- **No setup required**: All dependencies are handled automatically
- **Easy data upload**: Direct integration with Google Drive and file upload
- **Interactive environment**: Real-time monitoring and visualization
- **Collaboration**: Easy sharing and collaboration features

### Hardware Used
This project was trained on **Google Colab with T4 GPU**, which provides:
- 16GB GPU memory
- Sufficient for batch size of 16 with 640x640 images
- Training time: ~75 minutes for 50 epochs

## 📊 Dataset Preparation

### Data Structure
Your dataset should be organized in the following structure:
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    └── val/
        ├── image1.txt
        ├── image2.txt
        └── ...
```

### Data Configuration
Create a `data.yaml` file:
```yaml
path: /path/to/your/dataset
train: images/train
val: images/val

# Classes
names:
  0: plastic
  1: bottle
  2: bag
  # Add more classes as needed
```

## 🏋️ Training

### Quick Start Training
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.pt')  # Load pretrained model

# Train the model (optimized for T4 GPU)
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,  # Optimal for T4 GPU memory
    name='yolov8m_plastics_merged'
)
```

### Advanced Training Configuration
You can customize training parameters in the `args.yaml` file or pass them directly:

```python
results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,  # GPU device
    workers=8,
    patience=100,
    save_period=10,
    project='runs/detect',
    name='yolov8m_plastics_merged',
    exist_ok=True,
    pretrained=True,
    optimizer='auto',
    verbose=True,
    seed=0,
    deterministic=True,
    single_cls=False,
    amp=True,  # Automatic Mixed Precision
    fraction=1.0,
    cache=False,
    close_mosaic=10,
    resume=False,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    pose=12.0,
    kobj=1.0,
    nbs=64,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    auto_augment='randaugment',
    erasing=0.4,
    crop_fraction=1.0
)
```

### Training Monitoring
During training, you can monitor:
- Loss curves (box, classification, DFL)
- Precision and Recall metrics
- mAP (mean Average Precision) at different IoU thresholds
- Learning rate schedules

## 🔍 Inference

### Basic Inference
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('yolov8m_plastics_merged/weights/best.pt')

# Run inference on an image
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

### Batch Inference
```python
# Run inference on multiple images
results = model(['image1.jpg', 'image2.jpg', 'image3.jpg'])

# Process results
for result in results:
    print(f"Detected {len(result.boxes)} objects")
    result.show()
```

### Video Inference
```python
# Run inference on video
results = model('video.mp4', save=True)
```

### Custom Inference Settings
```python
results = model(
    'image.jpg',
    conf=0.25,        # Confidence threshold
    iou=0.7,          # NMS IoU threshold
    max_det=300,      # Maximum detections
    save=True,        # Save results
    save_txt=True,    # Save labels
    save_conf=True,   # Save confidences
    save_crop=True    # Save cropped predictions
)
```

## 📈 Results

Based on the training results, the model achieved:

- **Final mAP@0.5**: 0.8816 (88.16%)
- **Final mAP@0.5:0.95**: 0.64201 (64.20%)
- **Precision**: 0.88071 (88.07%)
- **Recall**: 0.80879 (80.88%)

### Training Progress
The model was trained for 50 epochs with the following key metrics:
- Box Loss: Decreased from 1.36 to 0.70
- Classification Loss: Decreased from 2.29 to 0.43
- DFL Loss: Decreased from 1.59 to 1.10

## 📁 Project Structure

```
Underwater Object Detection Group - 4/
├── README.md                           # This file
├── UWOD.ipynb                         # Main Jupyter notebook
├── report.pdf                         # Project report
├── yolov8m_plastics_merged/           # Training results
│   ├── args.yaml                      # Training arguments
│   ├── results.csv                    # Training metrics
│   ├── results.png                    # Training plots
│   ├── confusion_matrix.png           # Confusion matrix
│   ├── confusion_matrix_normalized.png # Normalized confusion matrix
│   ├── F1_curve.png                   # F1 score curve
│   ├── P_curve.png                    # Precision curve
│   ├── PR_curve.png                   # Precision-Recall curve
│   ├── R_curve.png                    # Recall curve
│   ├── labels.jpg                     # Label distribution
│   ├── labels_correlogram.jpg         # Label correlation
│   ├── train_batch*.jpg               # Training batch examples
│   ├── val_batch*_labels.jpg          # Validation labels
│   ├── val_batch*_pred.jpg            # Validation predictions
│   └── weights/                       # Model weights
│       ├── best.pt                    # Best model weights
│       └── last.pt                    # Last epoch weights
```

## 🛠️ Usage Examples

### Example 1: Quick Detection
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8m_plastics_merged/weights/best.pt')

# Detect objects in image
results = model('underwater_image.jpg', conf=0.3)

# Show results
results[0].show()
```

### Example 2: Save Results
```python
# Run inference and save results
results = model(
    'underwater_image.jpg',
    save=True,
    save_txt=True,
    save_conf=True,
    conf=0.25
)
```

### Example 3: Batch Processing
```python
import os

# Process all images in a folder
image_folder = 'underwater_images/'
results = model(image_folder, save=True)
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `batch=8` or `batch=4`
   - Reduce image size: `imgsz=416` or `imgsz=320`

2. **Google Colab Issues**
   - **Runtime disconnection**: Save your work frequently and use `!pip install` for dependencies
   - **GPU not available**: Go to Runtime → Change runtime type → Hardware accelerator → GPU
   - **T4 GPU memory**: With 16GB memory, batch size 16 works well for 640x640 images
   - **Memory issues**: Restart runtime and clear outputs if needed
   - **File upload limits**: Use Google Drive for large datasets

3. **Installation Issues**
   - Ensure Python version is 3.8+
   - Install PyTorch with CUDA support if using GPU
   - Use virtual environment to avoid conflicts

4. **Training Issues**
   - Check data.yaml format
   - Verify image and label paths
   - Ensure sufficient disk space for saving results

### Performance Optimization

1. **GPU Usage**
   - Use CUDA-compatible GPU for faster training
   - Enable mixed precision training with `amp=True`

2. **Memory Optimization**
   - Use appropriate batch size for your GPU
   - Enable caching with `cache=True` for faster training

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 👥 Contributors

This project was developed by Group 4 for underwater object detection research.

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Note**: This README provides a comprehensive guide for using the underwater object detection model. For specific implementation details, refer to the `UWOD.ipynb` notebook.
