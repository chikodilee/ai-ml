# %% [markdown]
# ## Assignment 1
# ### Dataset B

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sns.set_theme()

# %% [markdown]
# Load data into a dataframe and split into features and labels

# %%
q2_df = pd.read_csv(Path('./DataB.csv').as_posix(), index_col=0)
df_features = q2_df.loc[:, q2_df.columns != 'gnd']
df_labels = q2_df.loc[:, ['gnd']]

# %% [markdown]
# Make sure labels and features have values in the expected range
# - Features must have values from 0-255 (pixel intensity)
# - Labels must have velues from 0-4 (representing 5 classes)

# %%
def check_data_range_issues(df: pd.DataFrame, expected_min: float | int, expected_max: float | int):
    """Checks if the values for all columns in the dataframe are within the expected min and max value.
    Just some sanity-check to make sure the data is clean
    Prints message if features have values outside expected range

    Args:
        df (pd.DataFrame): DataFrame object to verify
        expected_min (float | int): minimum expected value
        expected_max (float | int): maximum expected value
    """
    min_value_issue_features = np.where(df.min() < expected_min)[0]
    max_value_issue_features = np.where(df.min() > expected_max)[0]

    if len(min_value_issue_features):
        print(f'Found {len(min_value_issue_features)} with min value < {expected_min}: {df.columns[min_value_issue_features].tolist()}')
    if len(max_value_issue_features):
        print(f'Found {len(max_value_issue_features)} with max value > {expected_max}: {df.columns[max_value_issue_features]}')

check_data_range_issues(df_features, expected_min=0, expected_max=255)
check_data_range_issues(df_labels, expected_min=0, expected_max=4)

# %%
# apply z-score normalization to features
df_features_z_norm = df_features.apply(lambda col: col - col.mean() / col.std())

# %% [markdown]
# #### 1. Use PCA as a dimensionality reduction technique to the data, compute the eigenvectors and eigenvalues.

# %%
# using scikit-learn library to perform Principal Component Analysis
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
pca = PCA()
pca = pca.fit(df_features_z_norm)
pca_transformed_features = pca.transform(df_features_z_norm)

eigenvectors = pca.components_
eigenvalues = pca.explained_variance_

# %%
def plot_components_by_class(x: np.ndarray, y: np.ndarray, class_labels: np.ndarray, x_label: str, y_label: str, title: str):
    """Helper function to plot data points for each class (using y_label) for the x and y components given

    Args:
        x (np.ndarray): data points along first component. shape must be (n_points, )
        y (np.ndarray): data points along second component. shape must be (n_points, )
        class_labels (np.ndarray): labels corresponding to each point. shape must be (n_points, )
        x_label (str): label for x axis
        y_label (str): label for y axis
        title (str): title for plot
    """
    unique_classes = class_labels.unique()

    for classification in unique_classes:
        plt.scatter(x[class_labels == classification], y[class_labels == classification], marker='.')
    plt.legend(unique_classes, title='Class', fancybox=True, bbox_to_anchor=(1.0, 0.5, 0.15, 0.3))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()

# %% [markdown]
# #### 2. Plot a 2-dimensional representation of the data points based on the first and second principal components. Explain the results versus the known classes (display data points of each class with a different color).

# %%
plot_components_by_class(
    x=pca_transformed_features[:, 0],
    y=pca_transformed_features[:, 1],
    class_labels=df_labels['gnd'],
    x_label=f'Principal component 1 (explained variance ratio: {round(pca.explained_variance_ratio_[0], 2)})',
    y_label=f'Principal component 2 (explained variance ratio: {round(pca.explained_variance_ratio_[1], 2)})',
    title='Class-wise data point distribution along first and second principal components'
    )
plt.show()

# %% [markdown]
# #### 3. Repeat step 2 for the 5th and 6st components. Comment on the result.

# %%
plot_components_by_class(
    y=pca_transformed_features[:, 4],
    x=pca_transformed_features[:, 5],
    class_labels=df_labels['gnd'],
    x_label=f'Principal component 5 (explained variance ratio: {round(pca.explained_variance_ratio_[4], 2)})',
    y_label=f'Principal component 6 (explained variance ratio: {round(pca.explained_variance_ratio_[5], 2)})',
    title='Class-wise data point distribution along fifth and sixth principal components'
    )
plt.show()

# %% [markdown]
# #### 4. Use the Naive Bayes classifier to classify 8 sets of dimensionality reduced data (using the first 2, 4, 10, 30, 60, 200, 500, and all 784 PCA components). Plot the classification error for the 8 sets against the retained variance of each case.

# %%
# using scikit-learn library to perform NaiveBayes classification
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#

num_components_of_interest = [2, 4, 10, 30, 60, 200, 500, 784]

# stores the (training) classification error for each naive bayes classifier using given components
train_classification_errors = []
# stores the cumulative percentage of variance explained by all the principal components used in classification
total_explained_variance = []

def calculate_classification_error(y_true: np.ndarray, y_pred: np.ndarray):
    """calculates classification error given true and predicted labels

    Args:
        y_true (np.ndarray): GT classification labels
        y_pred (np.ndarray): predicted classification labels

    Returns:
        float: value in [0, 1] representing classification error
    """
    return np.sum(y_true != y_pred) / np.shape(y_true)[0]

for num_components in num_components_of_interest:
    naive_bayes_classifier = GaussianNB().fit(pca_transformed_features[:, :num_components], df_labels['gnd'])
    predictions = naive_bayes_classifier.predict(pca_transformed_features[:, :num_components])
    train_classification_errors.append(calculate_classification_error(df_labels['gnd'], predictions))
    total_explained_variance.append(np.sum(pca.explained_variance_ratio_[:num_components]))

# plot results
legend = []
for i, num_components in enumerate(num_components_of_interest):
    plt.scatter(total_explained_variance[i], train_classification_errors[i])
    legend.append(f'{num_components} PCs')
plt.legend(legend, title='Number of PCs used', fancybox=False, bbox_to_anchor=(1.0, 0.5, 0.1, 0.3))
plt.ylabel('Training classification error')
plt.xlabel('Total % variance explained by principal components used')
plt.title('Naive bayes fitting error vs. Variance explained by principal components used')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### 5. As the class labels are already known, you can use the Linear Discriminant Analysis (LDA) to reduce the dimensionality, plot the data points using the first 2 LDA components (display data points of each class with a different color). Explain the results obtained in terms of the known classes. Compare with the results obtained by using PCA.

# %%
# using scikit-learn library to perform Linear Discriminant Analysis
# https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

lda = LinearDiscriminantAnalysis()
lda_components = lda.fit_transform(df_features_z_norm, df_labels['gnd'])
plot_components_by_class(
    x=lda_components[:, 0],
    y=lda_components[:, 1],
    class_labels=df_labels['gnd'],
    x_label=f'LDA component 1 (explained variance ratio: {round(lda.explained_variance_ratio_[0], 2)})',
    y_label=f'LDA component 2 (explained variance ratio: {round(lda.explained_variance_ratio_[1], 2)})',
    title='Class-wise data point distribution along first and second LDA components'
)
plt.show()

