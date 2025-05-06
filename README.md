# Wildfire Prediction with CNN & Swin Transformer

This project trains deep learning models to predict wildfire presence from satellite images. The goal was not to use the train labels in order to deal with a short dataset, trying to explore how to use the train data or other ressources in order to get a pretraining before finetuning on the specific task.

 It includes:
- A **baseline CNN** model only trained on the val dataset.
- A **Swin Transformer-based model** fine-tuned from a pretrained backbone.
- A **Vit model** with several size that can be trained with Dino before finetuning on the Val DATAset

---

## 1. Download Dataset
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

## 2. Download Pretrained Swin Transformer (Optional)
If training the **Swin Transformer**, download the pretrained weights:

```bash
wget https://huggingface.co/allenai/satlas-pretrain/resolve/main/satlas-model-v1-highres.pth -O satlas-model-v1-highres.pth
```

---


## 3. Train a Model

### Train the Baseline CNN
```bash
python train.py --args
```
### Pretrain the VIT
```bash
python train_dino_backbone.py --args
```
### Train the Swin Transformer
```bash
python train.py --args
```
---

## 4. Test the Model
Once trained, you can evaluate the model on the test set:
```bash
python train.py --model baseline --data_dir "./wildfire-prediction-dataset" --epochs 0
```

---
