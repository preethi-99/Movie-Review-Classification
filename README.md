Movie Review Classification
Overview
A K-Nearest Neighbors (KNN) classifier for movie sentiment analysis, trained on 25,000 reviews. Achieved 0.72 accuracy using TF-IDF vectorization and Principal Component Analysis (PCA) for feature reduction.

Dataset
Train_new.txt – Training dataset (labeled reviews).
Test_new.txt – Testing dataset (unlabeled reviews).
Format.txt – Reference file.
Workflow
Data Preprocessing – Tokenization, stopword & punctuation removal, lowercasing.
Feature Extraction – TF-IDF vectorization (5,000 features).
Dimensionality Reduction – PCA (50 components).
KNN Classification – Best accuracy (0.75 at k=12,14).
Execution (Google Colab)
Upload Train_new.txt, Test_new.txt, Format.txt.
Run the script/notebook.
Predictions are saved to the output file.
Results
Final Accuracy: 0.72
Optimized TF-IDF + PCA for efficiency
Future Work
Experiment with SVM, Logistic Regression, Deep Learning.
Use Word2Vec or BERT embeddings for better representation.
📧 Author: Preethi Ranganathan (prangana@gmu.edu)
