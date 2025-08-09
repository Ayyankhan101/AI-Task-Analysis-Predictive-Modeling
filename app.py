
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('AI_DATA.csv')
    return df

df = load_data()
df_clustered = df.copy()

st.title("AI Task Analysis and Predictive Modeling")

st.markdown("""
**Project Goal:** The objective of this project is to conduct a thorough analysis of the `AI_DATA.csv` dataset. We aim to uncover underlying patterns in AI-related tasks, group them into meaningful categories using machine learning, and build a model that can automatically classify new tasks.
""")

# Phase 1: Data Understanding and Exploratory Data Analysis (EDA)
st.header("Phase 1: Data Understanding and Exploratory Data Analysis (EDA)")

st.subheader("1.1. Initial Data Inspection")
st.write("First 5 rows of the dataset:")
st.write(df.head())
st.write("Dataset Info:")
st.text(df.info())
st.write("Missing Values:")
st.write(df.isnull().sum())
st.write("Descriptive Statistics:")
st.write(df.describe())

st.subheader("1.2. Visualizing Data Distributions")
numerical_features = ['feedback_loop', 'directive', 'task_iteration', 'validation', 'learning', 'filtered']
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
df[numerical_features].hist(bins=15, ax=axes)
plt.suptitle('Distribution of Numerical Features')
st.pyplot(fig)

st.subheader("1.3. Correlation Analysis")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
plt.title('Correlation Matrix of Numerical Features')
st.pyplot(fig)

# Phase 2: Text Analysis of task_name
st.header("Phase 2: Text Analysis of `task_name`")

st.subheader("2.1. Word Cloud")
text = ' '.join(df['task_name'].dropna())
wordcloud = WordCloud(width=1000, height=400, background_color='white').generate(text)
fig, ax = plt.subplots(figsize=(25, 5))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
plt.title('Word Cloud of Task Names')
st.pyplot(fig)

st.subheader("2.2. N-gram Analysis")
def get_top_ngrams(corpus, n=None, ngram_range=(1, 1)):
    vec = TfidfVectorizer(ngram_range=ngram_range, stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

top_unigrams = get_top_ngrams(df['task_name'].dropna(), n=20, ngram_range=(1, 1))
top_bigrams = get_top_ngrams(df['task_name'].dropna(), n=20, ngram_range=(2, 2))

st.write("Top 20 Unigrams:")
st.write(top_unigrams)
st.write("Top 20 Bigrams:")
st.write(top_bigrams)

# Phase 3: Unsupervised Learning - Clustering
st.header("Phase 3: Unsupervised Learning - Clustering")

st.subheader("3.1. Feature Scaling and Finding Optimal Clusters")
features = df[numerical_features].dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
st.pyplot(fig)

st.subheader("3.2. K-Means Clustering and Visualization")
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_clustered = df.dropna(subset=numerical_features).copy()
df_clustered['cluster'] = kmeans.fit_predict(scaled_features)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
df_clustered['pca1'] = principal_components[:, 0]
df_clustered['pca2'] = principal_components[:, 1]

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df_clustered, palette='viridis', s=100, alpha=0.7, ax=ax)
plt.title('Task Clusters (PCA)')
st.pyplot(fig)

# Phase 4: Supervised Learning - Task Classification
st.header("Phase 4: Supervised Learning - Task Classification")

st.subheader("4.1. Feature Engineering and Model Training")
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df_clustered['task_name'])
y = df_clustered['cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("4.2. Model Evaluation")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

st.write("Confusion Matrix:")
mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Confusion Matrix')
st.pyplot(fig)
