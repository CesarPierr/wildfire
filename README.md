# Wildfire Prediction with CNN & Swin Transformer

This project trains deep learning models to predict wildfire presence from satellite images. It includes:
- A **baseline CNN** model.
- A **Swin Transformer-based model** fine-tuned from a pretrained backbone.

## 📂 Project Structure
```
.
├── wildfire-prediction-dataset/  # Dataset (train, valid, test)
├── utils.py                      # Helper functions (dataset prep, training, validation, plotting)
├── models.py                     # Model architectures (CNN, Swin Transformer)
├── train.py                      # Main script for training
├── outputs/                       # Saved models & plots
└── README.md                      # Project documentation
```

---

## 📥 1. Download Dataset
The dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset):

```bash
# Download
kaggle datasets download -d abdelghaniaaba/wildfire-prediction-dataset
# or 
curl -L -o ./wildfire-prediction-dataset.zip https://www.kaggle.com/api/v1/datasets/download/abdelghaniaaba/wildfire-prediction-dataset
# Extract
unzip wildfire-prediction-dataset.zip -d wildfire-prediction-dataset
```

It should contain:
```
wildfire-prediction-dataset/
├── train/
├── valid/
└── test/
```

---

## 📥 2. Download Pretrained Swin Transformer (Optional)
If training the **Swin Transformer**, download the pretrained weights:

```bash
wget https://huggingface.co/allenai/satlas-pretrain/resolve/main/satlas-model-v1-highres.pth -O satlas-model-v1-highres.pth
```

---

## 🚀 3. Install Dependencies
```bash
pip install torch torchvision matplotlib tqdm scikit-learn
```
(If using GPU, install the CUDA-compatible version of PyTorch: [PyTorch Installation](https://pytorch.org/get-started/locally/))

---

## 🏋️ 4. Train a Model

### Train the Baseline CNN
```bash
python train.py \
    --model baseline \
    --data_dir "./wildfire-prediction-dataset" \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_amp
```

### Train the Swin Transformer
```bash
python train.py \
    --model swin \
    --checkpoint_swin "satlas-model-v1-highres.pth" \
    --data_dir "./wildfire-prediction-dataset" \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --use_amp
```

---

## 📊 5. Results & Checkpoints
- The best model checkpoint is saved automatically in `./outputs/`
- Training loss, validation loss, and accuracy curves are saved as a figure in `./outputs/`

---

## 📝 Notes
- `--use_amp` enables **Automatic Mixed Precision (AMP)** for faster training on GPUs.
- Adjust `--epochs`, `--batch_size`, `--lr` as needed.
- Ensure dataset structure matches expectations (`train/`, `valid/`, `test/`).

---

## 🔥 Test the Model
Once trained, you can evaluate the model on the test set:
```bash
python train.py --model baseline --data_dir "./wildfire-prediction-dataset" --epochs 0
```
(Change `--model baseline` to `swin` for the Swin Transformer.)

---
