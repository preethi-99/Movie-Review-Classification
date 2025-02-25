

import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn
from nltk.corpus import genesis
nltk.download('genesis')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
genesis_ic = wn.ic(genesis, False, 0.0)

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

columns = ['Sentiment', 'Text Review']

numbers = []
text =[]

# Read the plain text file line by line with a specified encoding
with open('Training.txt', 'r', encoding='utf-8') as file:

    for line in file:

         if len(line) >= 9:

             
             numbers.append(line[0:2])
             #large_data = line[3:12]  # Assuming the n columns of data start from the second column
             text.append(line[3:])
             #data.append([small_data,large_data])





train_df = pd.DataFrame({'Sentiment': numbers,
                     'Text': text})

print(train_df)

#display(df['Text Review'])

test_text = []

with open('Testing.txt', 'r', encoding='utf-8') as file:
    for line in file:
         if len(line) >= 9:
             #large_data = line[3:12]  # Assuming the n columns of data start from the second column
             test_text.append(line[:])
             #data.append([small_data,large_data])


test_df = pd.DataFrame({'Text': test_text})

print(test_df)

import re
nltk.download('stopwords')

import string
string.punctuation
wordnet_lemmatizer = WordNetLemmatizer()
def cleaning_data(text):
    words = nltk.word_tokenize(text)

    # Lowercasing
    words = [word.lower() for word in words]

    # Remove punctuation and special characters
    words = [word for word in words if word not in string.punctuation]

    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stopwords]

    #Lemmization
    words = [wordnet_lemmatizer.lemmatize(word) for word in words]

    # Join the tokens back into a single string
    cleaned_data = ' '.join(words)

    return cleaned_data

train_df['cleaned_data_reviews']= train_df['Text'].apply(cleaning_data)
print(train_df['cleaned_data_reviews'])

test_df['test_cleaned_data_reviews']= test_df['Text'].apply(cleaning_data)
print(test_df['test_cleaned_data_reviews'])

data = train_df['cleaned_data_reviews']
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

#Fitting and transforming the vectorizer on preprocessed train reviews
tfidf_matrix_train = tfidf_vectorizer.fit_transform(data)

print(tfidf_matrix_train)

data1 = test_df['test_cleaned_data_reviews']
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

#Fitting and transforming the vectorizer on preprocessed train reviews
tfidf_matrix_test = tfidf_vectorizer.fit_transform(data1)


print(tfidf_matrix_test)


pca = PCA(n_components=50)

# Fit and transform the PCA on the TF-IDF matrix
train_reduced_data = pca.fit_transform(tfidf_matrix_train.toarray())

from sklearn.decomposition import PCA

# Assuming tfidf_matrix_test is your sparse TF-IDF matrix

dense_tfidf_matrix_test = tfidf_matrix_test.toarray()

# Initialize PCA with the desired number of components (e.g., n_components=100)
pca = PCA(n_components=50)

# Fit and transform PCA on the dense TF-IDF matrix
test_reduced_data = pca.fit_transform(dense_tfidf_matrix_test)

X_train, X_val, y_train, y_val = train_test_split(train_reduced_data, train_df['Sentiment'], test_size=0.2, random_state=42)

X_test = test_reduced_data

#List of 'k' values to consider
values_of_k = list(range(1, 15))

#An empty list to store cross-validation scores
cross_validation_scores = []

# Performing k-fold cross-validation for each 'k' value
for k in values_of_k:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_classifier, X_train, y_train, scoring='accuracy') # performing 5-fold cross validation
    print(f"Cross-validation with k={k}: {scores.mean():.2f} accuracy")
    cross_validation_scores.append(np.mean(scores))

# Finding the 'k' value with the highest cross-validation score
best_k = values_of_k[cross_validation_scores.index(max(cross_validation_scores))]

# Print the best 'k' value and its corresponding cross-validation score
print("Best 'k' value:", best_k)
print("Cross-Validation Score (Accuracy):", max(cross_validation_scores))

from sklearn.neighbors import KNeighborsClassifier

# Create a k-NN classifier with the optimal 'k' value
knn_classifier = KNeighborsClassifier(n_neighbors=best_k)

# Train the classifier on the entire training dataset
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)

print(len(y_pred))

# Save the test predictions to a file
with open("predictions.txt", "w", encoding="utf-8") as file:
    for prediction in y_pred:
        file.write(f"{prediction}\n")

test_sentiments = []
with open("format.txt", "r", encoding="utf-8") as file:
    for line in file:
        test_sentiments.append(line.strip())
print(len(test_sentiments))

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt



# Accuracy
accuracy = accuracy_score(test_sentiments, y_pred)
print("Accuracy:", accuracy)

# Precision
precision_true = precision_score(test_sentiments, y_pred, pos_label='+1')  # Use pos_label=+1 for positive class
precision_false = precision_score(test_sentiments, y_pred, pos_label='-1')  # Use pos_label=-1 for negative class
print("Precision for (+1):", precision_true)
print("Precision  for (-1):", precision_false)

