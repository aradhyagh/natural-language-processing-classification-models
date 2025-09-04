# 📝 NLP Text Classification - Sentiment Analysis

This repository contains multiple Natural Language Processing (NLP) models built for **text sentiment classification**. The project experiments with different techniques including **Naive Bayes** and a **GRU-based Deep Learning model**, applied on COVID-19 related tweets dataset.

---

## 📂 Repository Structure

├── Corona_NLP_test (1).csv # Raw dataset

├── preprocessed_data_sentiment_classification.csv # Cleaned & preprocessed dataset

├── sentiment_classification_project (1).ipynb # Main Jupyter Notebook

├── GRUmodel.h5 # Trained GRU deep learning model

├── naive_bayes_model (1).pkl # Saved Naive Bayes model

├── count_vectorizer_for_naivebayse.pkl # CountVectorizer used in Naive Bayes

├── label_encoder_sentiment.pkl # Label Encoder for target sentiments

├── README.md # Documentation file

---

## 📊 Project Overview

The goal of this project is to classify text data into different **sentiment categories** (e.g., Positive, Negative, Neutral) using both **classical ML methods** and **Deep Learning methods**.

- **Dataset**: Corona_NLP dataset (COVID-19 tweets with sentiment labels)  
- **Task**: Sentiment classification  
- **Models Used**:
  - Naive Bayes with CountVectorizer  
  - GRU (Gated Recurrent Unit) Neural Network  

---

## ⚙️ Features & Workflow

1. **Data Preprocessing**  
   - Cleaning tweets (removing stopwords, punctuation, special characters)  
   - Tokenization & Lemmatization  
   - Handling class imbalance  

2. **Feature Engineering**  
   - CountVectorizer (for Naive Bayes)  
   - Tokenizer + Padding (for GRU model)  

3. **Models Implemented**  
   - **Naive Bayes**: A simple but effective baseline model  
   - **GRU Model**: Deep learning architecture for sequential data  

4. **Evaluation Metrics**  
   - Accuracy  
   - F1 Score  
   - Confusion Matrix  

---

## 📈 Results

- **Naive Bayes Model**: Achieved reasonable baseline performance  
- **GRU Model**: Outperformed traditional ML methods in accuracy and F1 score  

*(Add actual results/accuracy values from your notebook here)*

---

## 🛠️ Tech Stack

- Python  
- NumPy, Pandas, Scikit-learn  
- TensorFlow / Keras  
- Matplotlib, Seaborn  

---

## 📌 Future Improvements

- Experiment with LSTM and Bidirectional GRU  
- Fine-tune transformer models (BERT, RoBERTa)  
- Deploy the model using Streamlit / Flask  

---

## 📬 Contact

👤 **Aradhya Phutak**  
- 📍 Mumbai, Maharashtra  
- 💼 Data Analyst & Machine Learning Engineer  
- 🔗 [LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/aradhyagh)  
