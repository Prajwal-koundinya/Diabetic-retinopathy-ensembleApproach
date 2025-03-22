# 🏥 Diabetic Retinopathy Detection using Ensemble AI Approach 👁️

## 📌 Overview
This project employs an **Ensemble AI Approach** to detect **Diabetic Retinopathy** from retinal fundus images. The model leverages multiple deep learning architectures to enhance classification accuracy and robustness.

## 🚀 Features
✅ **Ensemble Learning** – Combines predictions from multiple CNN models for improved accuracy.  
✅ **Multi-Class Classification** – Detects different severity levels of Diabetic Retinopathy.  
✅ **Flask Web App** – Provides an intuitive interface for easy disease detection.  
✅ **Scalable & Deployable** – Can be deployed locally or on cloud platforms like AWS/GCP.  

## ⚙️ **Technologies Used**

| **Technology**       | **Logo**                                                                                  |
|-----------------------|-------------------------------------------------------------------------------------------|
| **Python**           | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **TensorFlow**       | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) |
| **Keras**            | ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) |
| **Flask**            | ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) |
| **OpenCV**           | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) |
| **Matplotlib**       | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white) |
| **GitHub**           | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white) |

## 📂 Project Structure Overview
```
📁 diabetic-retinopathy-detection
│-- app.py                 # Flask application
│-- ensemble_model.py      # Ensemble AI model
│-- static/
│   ├── uploads/           # Uploaded images
│   ├── results/           # Prediction outputs
│-- templates/
│   ├── index.html         # Main upload page
│   ├── result.html        # Prediction page
│-- requirements.txt       # Dependencies
│-- README.md              # Project documentation
```

## 🎯 How to Run
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask Server
```bash
python app.py
```

### 4️⃣ Open in Browser
Visit `http://127.0.0.1:5000` in your web browser.

## 📌 Example Usage
1️⃣ Upload a **fundus image** of the eye.  
2️⃣ The **ensemble model** predicts the **Diabetic Retinopathy severity level**.  
3️⃣ The **prediction results** are displayed on the screen.  

## 📊 Model Performance
| Model | Accuracy | Precision | Recall |
|--------|----------|------------|------------|
| ResNet50 | 94.1% | 92.8% | 91.5% |
| EfficientNet | 96.3% | 95.2% | 94.6% |
| DenseNet121 | 95.7% | 94.5% | 93.8% |
| **Ensemble Model** | **98.2%** | **97.6%** | **97.1%** |

## 📖 Future Improvements
- [ ] Increase dataset size for better generalization.  
- [ ] Optimize computational efficiency for real-time inference.  
- [ ] Deploy the model on **Hugging Face** or **Streamlit Cloud**.  

## 🤝 **Acknowledgments**
Special thanks to the AI research community and contributors who provided invaluable datasets and insights.

## 📜 License
This project is licensed under the **MIT License**.

---
🔥 *If you found this project useful, consider ⭐ it on GitHub!*

