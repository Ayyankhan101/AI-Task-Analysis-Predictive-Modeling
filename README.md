# AI Task Analysis and Predictive Modeling

## Project Overview

This project performs a comprehensive data analysis on the `AI_DATA.csv` dataset. The primary goal is to understand the characteristics of various AI-related tasks, uncover hidden patterns through unsupervised machine learning, and build a predictive model to classify tasks based on their textual descriptions. The entire analysis is documented and presented in the `ai_task_analysis.ipynb` Jupyter Notebook.

## Dataset

The dataset used is `AI_DATA.csv`, which contains the following columns:

*   `task_name`: A textual description of the task.
*   `feedback_loop`, `directive`, `task_iteration`, `validation`, `learning`, `filtered`: Numerical features representing different characteristics of the tasks.

## Methodology

The analysis is divided into four main phases:

1.  **Exploratory Data Analysis (EDA):** Initial inspection of the data, including statistical summaries and visualizations to understand the distribution and correlation of features.
2.  **Text Analysis:** In-depth analysis of the `task_name` column using Word Clouds and N-gram analysis to identify common themes and phrases.
3.  **Unsupervised Learning (Clustering):** Using the K-Means clustering algorithm to group similar tasks based on their numerical features. The optimal number of clusters is determined using the Elbow Method.
4.  **Supervised Learning (Classification):** Building a Multinomial Naive Bayes classifier to predict the task cluster (task type) based on the `task_name`. The model is trained and evaluated for performance.

## Installation

To run this project, you need to have Python 3 and the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn jupyter
```

## Usage

To view and run the analysis, launch the Jupyter Notebook:

```bash
jupyter notebook ai_task_analysis.ipynb
```

This will open the notebook in your web browser. You can then execute the cells sequentially to reproduce the analysis and see the results.

## Results and Findings

### Exploratory Data Analysis (EDA)

*   The dataset was found to be clean with no missing values.
*   The numerical features (`feedback_loop`, `directive`, `task_iteration`, `validation`, `learning`, `filtered`) exhibit a variety of distributions, with most being right-skewed.
*   The correlation heatmap revealed some interesting relationships between the numerical features, helping to understand how different task characteristics relate to each other.

### Text Analysis

*   **Word Cloud:** The most prominent words in the task descriptions include "prepare," "information," "develop," "data," and "reports."
*   **N-gram Analysis:**
    *   **Top Unigrams:** "prepare", "information", "develop", "data", "reports".
    *   **Top Bigrams:** "prepare reports", "provide information", "develop implement".

### Unsupervised Learning (Clustering)

*   The Elbow method suggested that the optimal number of clusters for this dataset is **4**.
*   K-Means clustering was successfully applied to group the tasks into these four distinct clusters based on their numerical features.
*   PCA was used to reduce the dimensionality of the features, and a scatter plot of the two principal components showed a reasonable separation of the clusters.

### Supervised Learning (Classification)

*   A Multinomial Naive Bayes classifier was trained to predict the task cluster (i.e., the task type) from the `task_name`.
*   The model achieved an accuracy of **50.67%** on the test set.
*   The classification report shows that the model performs differently for each cluster, with precision and recall values varying across the four classes. The confusion matrix visually represents the model's prediction accuracy for each cluster.

## Conclusion

This project successfully analyzed the `AI_DATA.csv` dataset, uncovering key characteristics of AI-related tasks. Through clustering, we identified four distinct types of tasks within the data. Furthermore, we built a predictive model that can classify new tasks with an accuracy of over 50%, which is a good starting point for a baseline model. The analysis provides valuable insights into the nature of AI tasks and demonstrates the power of combining NLP and clustering techniques for task categorization.
