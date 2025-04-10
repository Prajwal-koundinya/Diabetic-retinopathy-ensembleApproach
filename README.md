# 🏥 Diabetic Retinopathy Detection using Ensemble AI Approach 🔬

## 📌 Overview
This project leverages an **Ensemble Approach** for detecting **Diabetic Retinopathy** (DR) from retinal images. By combining multiple **pretrained CNN models**, the system enhances diagnostic accuracy and robustness for detection of DR. 

## 🚀 Features:
✅ **Ensemble Model** – Utilizes multiple CNN architectures for superior performance.  
✅ **High Accuracy** – Optimized for real-world clinical settings.  
✅ **Flask Web App** – User-friendly interface for easy image uploads and predictions.  
✅ **Scalable Deployment** – Can be hosted locally or on cloud platforms.  

## 🖼️ Model Performance
![image](https://github.com/Prajwal-koundinya/diabetic-retinopathy-detection/assets/model-performance.png)

*Above: An example of the ensemble model's performance on test data.*

## ⚙️ **Technologies Used**

| **Technology**       | **Logo**                                                                                  |
|-----------------------|-------------------------------------------------------------------------------------------|
| **Python**           | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **PyTorch**          | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) |
| **TensorFlow**       | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) |
| **Flask**            | ![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white) |
| **OpenCV**           | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) |
| **Matplotlib**       | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white) |

## 📂 Project Structure
```
📁 diabetic-retinopathy-detection
│-- app.py                 # Flask application
│-- model.py               # Ensemble AI model (CNN-based)
│-- static/
│   ├── uploads/           # Uploaded images
│   ├── results/           # Processed images
│-- templates/
│   ├── index.html         # Main upload page
│   ├── result.html        # Prediction page
│-- requirements.txt       # Dependencies
│-- README.md              # Project documentation
```

## 🎯 How to Run
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Prajwal-koundinya/diabetic-retinopathy-detection.git
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
1️⃣ Upload an **eye fundus image**.  
2️⃣ The ensemble model predicts the **Diabetic Retinopathy stage**.  
3️⃣ Results are displayed with confidence scores.  

## 🎨 UI Preview
![image](https://github.com/Prajwal-koundinya/diabetic-retinopathy-detection/assets/ui-preview.png)

## 📊 Results
| Retinopathy Stage | Model Prediction | Confidence |
|-------------------|-----------------|------------|
| Mild DR          | ✅ Correct       | 94.7% |
| Proliferative DR | ✅ Correct       | 96.3% |

## 📖 Future Improvements
- [ ] Improve dataset augmentation for better generalization.
- [ ] Optimize inference speed for real-time applications.
- [ ] Deploy on cloud platforms like AWS/Google Cloud.

## 🤝 **Acknowledgments**
Special thanks to the medical and AI communities for their valuable datasets and research.  
Inspirational guidance from **Dr. Victor Ikechukwu**. Explore their work: [Dr. Victor Ikechukwu](https://github.com/Victor-Ikechukwu). 

## 📜 License
This project is licensed under the **MIT License**.

---
🔥 *If you like this project, don't forget to ⭐ it on GitHub!*

