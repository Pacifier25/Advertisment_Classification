# Advertisement Classification System

## 🌟 Overview

The **Advertisement Classification System** leverages advanced AI technologies to classify advertisements into predefined categories based on textual content. By utilizing the **BERT** model (Bidirectional Encoder Representations from Transformers), the system ensures high accuracy in understanding and categorizing advertisement texts. 

With the capability for both image upload and real-time classification, the system is easy to use and highly effective. It assists in better understanding advertisement content and helps organizations in organizing their advertisements efficiently.

### 📊 Dataset

The dataset for this project consists of text data from various advertisements, which are classified into different categories. The goal is to classify the text into relevant categories based on the content.

### 🧠 Model

This system uses **BERT** (Bidirectional Encoder Representations from Transformers), a pre-trained model developed by Google, fine-tuned for the task of advertisement classification. It has shown outstanding performance in understanding the context of text.

**BERT** is fine-tuned using the advertisement dataset to categorize the content into labels. This model, known for its deep understanding of language, enhances the accuracy and precision of the classification.

### 🏅 Performance

- **Training Accuracy**: 99.81%
- **Validation Accuracy**: 99.90%

---

## 📊 Visualizations

### **Training vs Validation Loss**
![image](https://github.com/user-attachments/assets/b2893b0b-f4a0-4896-bdfb-339abfdb1ced)


### **Training vs Validation Accuracy**
![image](https://github.com/user-attachments/assets/6d2c1db0-3d83-47bb-a601-b1429f4d3fcb)


---

## 🎯 Features

- **Text Classification**: Classifies advertisement text based on predefined categories.
- **AI-Powered**: Uses deep learning (BERT model) to predict the category of advertisement text.
- **Real-Time Feedback**: Displays the classification result and confidence percentage instantly.
- **User-Friendly Interface**: Built with **Streamlit**, providing an easy-to-use platform for users.

---

## 🧑‍🏫 Classes

The dataset contains several predefined categories that advertisements can belong to. Here are the main categories:

- **Travel**: Ads related to travel destinations, travel packages, airlines, hotels, and vacation deals.
- **Science and Technology**: Ads related to technology, gadgets, scientific products, innovations, and technological services.
- **Food**: Ads promoting food products, fast food, restaurants, and food delivery services.
- **Manufacturing**: Ads related to industrial products, machinery, manufacturing companies, and services.
- **History**: Ads related to historical events, documentaries, museums, and cultural heritage.
- **Art and Music**: Ads promoting music albums, art galleries, musical performances, and artistic events.

These classes are used to categorize advertisements into specific groups based on their content, helping users to easily identify the subject of the advertisement.

---

## 🛠️ Technology Stack

- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow/Keras  
- **Natural Language Processing (NLP)**: BERT  
- **Frontend Framework**: Streamlit  
- **Deployment**: GitHub (currently), future deployment on Microsoft Azure

---

## 📂 Folder Structure
```
├── app.py 
├── requirements.txt 
├── data
│   └── Videos_data.csv
├── model
│   └── advertisement_classification_model.h5
└── README.md
```

## 📊 Results

- **Final Training Accuracy**: **99.81%**  
- **Final Validation Accuracy**: **98.62%**  
- **Model Type**: BERT (fine-tuned for advertisement text classification)  
- **Dataset Used**: Advertisement dataset containing text data

---

## 🎥 Demo Video
[Watch Demo Video](https://drive.google.com/file/d/1XBXA0wtbkkW88lU5drZ4z9-CepL-8ccW/view?usp=sharing)

---

## 💻 Installation Guide

### Prerequisites
- **Python** 3.7 or later  
- **Virtual Environment** (optional)

### Steps
1. **Clone the Repository**:
```bash
git clone https://github.com/Pacifier25/Advertisement-Classification.git
cd Advertisement-Classification

```
2. **Create Virtual Environment**:
```bash
Copy code
python -m venv env
source env/bin/activate      # For Linux/Mac
env\Scripts\activate         # For Windows
```

3. **Install Dependencies**:
```bash
pip install -r project_files/requirements.txt
```

4. **Run the Application**:
```bash
streamlit run app.py
```
---

## 🚀 Deployment
The application is deployed using Streamlit for easy web access. Future plans include deploying the project on Microsoft Azure for better scalability.

---

## 📥 Download
**[Click here to download the project files](https://github.com/Pacifier25/Waste-Classification-System/archive/refs/heads/main.zip)**

---
## 🙏 Acknowledgements
- **Model**: BERT (Bidirectional Encoder Representations from Transformers), pre-trained model from Hugging Face.
- **Dataset**: Custom dataset for advertisement classification.
- **Frameworks**: TensorFlow, Streamlit

---
