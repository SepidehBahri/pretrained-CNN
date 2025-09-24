#  Pretrained CNN Image Classifier

This project is a modular, general-purpose pipeline for image classification using **pretrained CNNs** with **transfer learning** in TensorFlow/Keras.

---

##  Features

- Plug-and-play with **any folder-based image dataset**
- Supports **InceptionResNetV2, ResNet50, EfficientNet, DenseNet**, and more
- Customizable **data augmentation** and **layer freezing**
- Automatically logs metrics to JSON
- Generates:
  -  Accuracy & loss curves
  -  Confusion matrix
- Fully configurable via `config.py`

---

## 🗂️ Project Structure

cnn-transfer-learning/ ├── data/ # Your dataset (structured as class subfolders) ├── models/ # Saved model files ├── logs/ # Training logs (JSON) ├── plots/ # Output figures ├── src/ # Core source code │ ├── config.py │ ├── data_loader.py │ ├── model_builder.py │ ├── train.py │ └── evaluate.py ├── main.py # CLI to train or evaluate ├── requirements.txt └── README.md

yaml
Copy
Edit

---

##  Dataset Format

Organize your dataset like this:

data/ ├── class1/ │ ├── img001.jpg │ └── ... ├── class2/ │ ├── img002.jpg │ └── ...

yaml
Copy
Edit

> It auto-detects classes based on folder names.

---

## Configuration

All hyperparameters and paths live in [`src/config.py`](src/config.py):

- Model selection (`InceptionResNetV2`, `ResNet50`, etc.)
- Input image size
- Batch size, learning rate, optimizer
- Data augmentation options
- Logging and output directories

---

##  Installation

```bash
git clone https://github.com/yourusername/cnn-transfer-learning.git
cd cnn-transfer-learning
pip install -r requirements.txt
 Training
bash
Copy
Edit
python main.py --mode train
The model and logs will be saved under models/ and logs/.

 Evaluation
bash
Copy
Edit
python main.py --mode evaluate
Generates:

Confusion matrix (plots/*.png)

