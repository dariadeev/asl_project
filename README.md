# 🤟 ASL Letter Recognition with Deep Learning

This repository presents a complete pipeline for recognizing American Sign Language (ASL) letters from static images using deep learning. It features a training script for model development and a Streamlit-based web interface for image-based inference.

---

## 🚀 Project Highlights

- 🔠 **Objective**: Classify static ASL hand gestures (A–Z, space, nothing, delete).
- 💬 **Motivation**: Improve communication access for Deaf and hard-of-hearing individuals while reducing reliance on human interpreters.
- 🧠 **Model**: Fine-tuned ResNet18 using PyTorch Lightning.
- 🌐 **Interface**: Streamlit app with image upload and real-time prediction.
- 🎯 **Accuracy**: Achieved 98.5% validation accuracy.

---

## 📁 File Structure

```
├── main.py                  # Main training script to generate the model checkpoint
├── app.py                   # Streamlit app for inference
├── asl_resnet_model.pt      # Trained model (TorchScript)
├── requirements.txt         # Dependencies for running the Streamlit app
├── kaggle.json              # Kaggle API key for dataset access
└── README.md                # Project documentation
```

---

## 🧪 Dataset

- **Dataset**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Contents**: 87,000+ labeled images of 29 ASL signs (A-Z, plus 3 control signs)
- **Augmentations**: Random crop, flip, jitter, and erasing

---

## 🔧 Tools & Technologies

- **Training**: Google Colab, PyTorch Lightning, torchvision
- **Model Architecture**: ResNet18
- **Monitoring**: Weights & Biases (WandB)
- **Deployment**: Streamlit

---

## 💻 How to Run

1. **Install requirements** for the Streamlit app:

```bash
pip install -r requirements.txt
```

2. **Start the app**:

```bash
streamlit run app.py
```

3. **Use the interface** to upload images and view predictions with confidence scores.
https://aslproject-hewqfyn6ontzsmxsge99by.streamlit.app/
---

## 📊 Performance

- ✅ High accuracy on distinct letters like A, B, C
- 🔁 Generalizes across lighting and hand shape variations
- ⚠️ Occasional confusion with visually similar signs (e.g., M/N, V/U, E/S)

---

## 🧱 Development Process

1. **Problem Definition**: Create a model for real-time ASL letter recognition.
2. **Model Selection**: ResNet-18 fine-tuned on ImageNet weights.
3. **Training Strategy**: Gradual unfreezing, data augmentation.
4. **Deployment**: Streamlit integration and image upload interface.

---

## 🔮 Future Plans

- 🖐️ Dynamic gesture recognition (videos, full signs)
- 📷 Real-time webcam support
- 🧾 Full sentence interpretation
- 📱 Mobile application deployment

---

## 🙏 Acknowledgments

- Kaggle for the ASL dataset
- PyTorch Lightning & Streamlit communities
- Weights & Biases for experiment tracking
