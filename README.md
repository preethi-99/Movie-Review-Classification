# **Movie Review Classification**

## **Overview**
This project implements a **K-Nearest Neighbors (KNN) classifier** for **movie sentiment analysis**, trained on **25,000 reviews**. The model achieves **0.72 accuracy** using **TF-IDF vectorization** and **Principal Component Analysis (PCA)** for feature reduction.

---

## **Dataset**
- 📂 **Train_new.txt** – Training dataset (labeled reviews)  
- 📂 **Test_new.txt** – Testing dataset (unlabeled reviews)  
- 📂 **Format.txt** – Reference file  

---

## **Workflow**
### **1. Data Preprocessing**
- ✅ Tokenization
- ✅ Stopword & punctuation removal
- ✅ Lowercasing

### **2. Feature Extraction**
- 🔍 **TF-IDF vectorization** (5,000 features)

### **3. Dimensionality Reduction**
- 📉 **PCA (50 components)** for improved efficiency

### **4. KNN Classification**
- 🎯 Best accuracy (**0.75 at k=12, 14**)

---

## **Execution (Google Colab)**
1. 📥 **Upload** `Train_new.txt`, `Test_new.txt`, `Format.txt` to **Google Colab**.  
2. 🏃 **Run the script or Jupyter Notebook**.  
3. 📊 **Predictions** are saved to the output file.  

---

## **Results**
- 🏆 **Final Accuracy:** `0.72`  
- ⚡ Optimized **TF-IDF + PCA** for efficiency  

---

## **Future Work**
- 🔬 Experiment with **SVM, Logistic Regression, Deep Learning**  
- 🤖 Use **Word2Vec or BERT embeddings** for better representation  

---

## **Author**
👤 **Preethi Ranganathan**  
📧 [prangana@gmu.edu](mailto:prangana@gmu.edu)  

---



