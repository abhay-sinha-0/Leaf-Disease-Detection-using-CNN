# 🌿 Leaf Disease Detection using CNN

![Plant Disease](https://miro.medium.com/v2/resize:fit:720/format:webp/1*D4B_s7_0rJ8TDBabmfrOnA.png)

## 📌 Objective
This project aims to detect and classify plant leaf diseases using Convolutional Neural Networks (CNNs). Early detection of diseases can help farmers take preventive action and improve crop yield. This solution is built using TensorFlow/Keras and is trained on the PlantVillage dataset.

---

## 🧰 Technologies Used

- 🧠 Deep Learning: TensorFlow & Keras (CNN)
- 🖼️ Image Processing: OpenCV, ImageDataGenerator
- 📊 Evaluation: Scikit-learn, Matplotlib
- 🐍 Language: Python

---

## 🗂️ Dataset

- **Source**: [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Structure**:
dataset/
├── train/
│ ├── healthy/
│ └── diseased/
└── test/
├── healthy/
└── diseased/

yaml
Copy
Edit

You can modify the class folder names and number of disease categories as needed.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/leaf-disease-detection.git
cd leaf-disease-detection
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Organize Dataset
Place your dataset inside the dataset/ folder following the structure mentioned above.

4️⃣ Run the Project
bash
Copy
Edit
python main.py
📈 Model Architecture
Input Image: 128x128x3

3 Convolutional Layers (32 → 64 → 128 filters)

MaxPooling after each Conv layer

Flatten → Dense Layer (256 neurons) → Dropout

Output Layer: Softmax for multi-class classification

🔍 Outputs
Accuracy and Loss plots for training/validation

Classification Report showing precision, recall, f1-score

Trained model saved to: model/leaf_disease_model.h5

📊 Sample Results
plaintext
Copy
Edit
Test Accuracy: 94.26%

Classification Report:
              precision    recall  f1-score   support

     diseased       0.93      0.95      0.94       210
      healthy       0.95      0.93      0.94       190

    accuracy                           0.94       400
💾 Saved Model
The trained CNN model is saved in the /model directory:

Copy
Edit
model/
  └── leaf_disease_model.h5
You can use this model for prediction or deploy it via Flask/Streamlit.

🚀 Future Work
Deploy model via web app (Streamlit/Flask)

Convert to TensorFlow Lite for mobile apps

Integrate Grad-CAM to visualize disease-affected regions

Expand dataset to include more disease categories

🧑‍💻 Author
Abhay Kumar
B.Tech CSE | Data Science & Machine Learning Enthusiast

