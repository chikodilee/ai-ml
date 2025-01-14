# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay

random_seed = 42

# %%
# read data from csv
df = pd.read_csv('./data/DataB1.csv')
df_features = df[['sepallength','sepalwidth','petallength','petalwidth']]
df_labels = df[['class']]

# split into train and test datasets
# Using stratify here to ensure class proportion is maintained in train and test datasets
train_features, test_features, train_labels, test_labels = train_test_split(
    df_features,
    df_labels,
    stratify=df['class'],
    test_size=0.3,
    random_state=random_seed,
    shuffle=True
)

# %%
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """function to calculate the class-wise accuracy, precision, recall, and f1 score

    Args:
        y_true (np.ndarray): 1-D array containing true class labels
        y_pred (np.ndarray): 1-D array containing predicted class labels

    Returns:
        pd.DataFrame: DataFrame with class-wise accuracy, precision, recall, and f1 score
    """
    metrics = {}
    classifications = np.unique(y_true)
    for classification in classifications:
        tp = np.logical_and(y_true == classification, y_pred == classification).sum()
        tn = np.logical_and(y_true != classification, y_pred != classification).sum()
        fp = np.logical_and(y_true != classification, y_pred == classification).sum()
        fn = np.logical_and(y_true == classification, y_pred != classification).sum()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2. * tp) / (2. * tp + fp + fn)

        metrics['accuracy'] = {**{classification: accuracy}, **metrics.get('accuracy', {})}
        metrics['precision'] = {**{classification: precision}, **metrics.get('precision', {})}
        metrics['recall'] = {**{classification: recall}, **metrics.get('recall', {})}
        metrics['f1'] = {**{classification: f1}, **metrics.get('f1', {})}

    metrics_df = pd.DataFrame.from_dict(metrics)
    metrics_df.loc['mean'] = metrics_df.mean()

    return metrics_df

def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Computes the confusion matrix for the given true and predicted labels

    Args:
        y_true (np.ndarray): 1-D array containing true class labels
        y_pred (np.ndarray): 1-D array containing predicted class labels

    Returns:
        pd.DataFrame: dataframe with confusion matrix. Column names and indices correspond to class labels
    """
    classes = np.unique(np.concatenate((y_true, y_pred)))

    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.uint32)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[classes == t, classes == p] += 1

    return pd.DataFrame(confusion_matrix, columns=classes).set_index(classes)

def pretty_print_results(y_true, y_pred, title):
    """prints accuracy, precision, recall, f1 score, and confusion matrix

    Args:
        y_true (np.ndarray): 1-D array containing true class labels
        y_pred (np.ndarray): 1-D array containing predicted class labels
    """
    print('-'*len(f'metrics for {title}'))
    print(f'Matrics for {title}')
    print('-'*len(f'metrics for {title}'))
    print(calculate_metrics(y_true, y_pred))
    print('\n')
    confusion_matrix = calculate_confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(confusion_matrix.to_numpy(), display_labels=confusion_matrix.columns).plot()
    plt.show()

# %% [markdown]
# To implement the SVM multi-class classifiers, we use the linear SVM from scikit-learn.

# %% [markdown]
# SVM - one-vs-one classifier
# The one-vs-one classification method trains multiple models, where each model is responsible to distinguish between a class pair. The outputs of each of the models are aggregated using class voting. This approach created c * (c - 1) / 2 classifiers, where c is the number of classes in the training labels.
# In this example of iris dataset with 3 classes, this strategy will train 3 models:
# - Iris-virginica vs Iris-versicolor
# - Iris-virginica vs Iris-setosa
# - Iris-versicolor vs Iris-setosa
# 
# This method can be time consuming when the number of classes is large.

# %%
class OVOClassifier:
    """Implementation of one-vs-one classifier for multi-class classification using SVM with linear kernel
    """
    def __init__(self) -> None:
        self._models = []
        self._classes = []

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Training function to generate and fit models for one-vs-one multiclass classification
        This trains n * (n - 1) / 2 classifiers if n > 1 and 1 otherwise.
        n represents the number of unique classses in y

        Args:
            x (np.ndarray): training features of shape (n_samples, n_features)
            y (np.ndarray): training labels of shape (n_samples, )

        Returns:
            OVOClassifier: reference to self with trained models
        """
        self._classes = np.unique(y)

        for i, class_1_label in enumerate(self._classes):
            for j, class_2_label in enumerate(self._classes):
                if j <= i:
                    continue
                indices_of_interest = (y == class_1_label) | (y == class_2_label)
                data_for_model = x[indices_of_interest]
                labels_for_model = y[indices_of_interest]
                model = SVC(kernel='linear', random_state=random_seed).fit(data_for_model, labels_for_model)

                self._models.append(model)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Inference function using one-vs-one training strategy.
        Predictions are resolved using voting and the class with the highest vote is selected as the output.
        In cases with ties (all classes are predicted), the class furthest from it's decision boundary is selected
        Requires fit() function to be called prior to calling this

        Args:
            x (np.ndarray): Features of samples to predict on. Expected shape is (n_samples, n_features)

        Returns:
            np.ndarray: 1D array containing predictions. Shape is (n_samples, )
        """
        predictions = [model.predict(x) for model in self._models]
        prediction_scores = [model.decision_function(x) for model in self._models]

        resolved_predictions = []
        for i in range(len(x)):
            predictions_for_index = [predictions[j][i] for j in range(len(self._models))]
            unique_preds, counts = np.unique(predictions_for_index, return_counts=True)

            # edge case where all classes are predicted
            if np.max(counts) == 1:
                # break tie using decision-function
                # documentation suggests that positive values indicate score for last class in model.classes_
                # https://scikit-learn.org/stable/glossary.html#term-decision_function
                decision_boundary_vals_for_index = [
                    (prediction_scores[j][i] if model.classes_[-1] == predictions_for_index[j] else prediction_scores[j][i] * -1)
                    for j, model in enumerate(self._models)
                ]
                best_val = np.argmax(decision_boundary_vals_for_index)
                resolved_prediction = predictions_for_index[best_val]
            else:
                highest_vote = np.argmax(counts)
                resolved_prediction = unique_preds[highest_vote]
            resolved_predictions.append(resolved_prediction)

        return np.array(resolved_predictions)

# %% [markdown]
# SVM - one-vs-all (one-vs-rest) classifier
# The one-vs-all classification method trains c classifiers, where c is the number of classes in the training labels. Each classifier determines if the features correspond to it's positive class or not.
# In this example of iris dataset with 3 classes, this strategy will train 3 models:
# - Iris-virginica vs others
# - Iris-setosa vs others
# - Iris-versicolor vs others
# 
# To aggregate the labels during inference, the predictions from each model are checked to see if they predicted their positive class (not 'other'). This can lead to situations where multiple models predict the positive class. In this implementation, such scenarios are dealt with by selecting the class that is the furthest from it's corresponding decision boundary.
# This approach can also lead to cases where all the models predict 'other'. In such cases, the positive class where the 'other' prediction is closest to the decision boundary is selected as the resolved prediction.
# 
# This method can be faster than the OVOClassifier, but can be sensitive to class imbalances and lead to ambiguious predictions.

# %%
class OVAClassifier:
    """Implementation of one-vs-all (or one-vs-rest) classifier for multi-class classification using SVM with linear kernel
    """
    def __init__(self) -> None:
        self._models = []
        self._classes = []
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Training function to generate and fit models for one-vs-all multiclass classification
        n classifiers will be generated, where n is the number of unique classes in y

        Args:
            x (np.ndarray): training features of shape (n_samples, n_features)
            y (np.ndarray): training labels of shape (n_samples, )

        Returns:
            OVAClassifier: reference to self with trained models
        """
        self._classes = np.unique(y)

        for class_label in self._classes:
            labels_for_model = np.copy(y)
            labels_for_model[labels_for_model != class_label] = 'other'

            model = SVC(kernel='linear', random_state=random_seed).fit(x, labels_for_model)
            self._models.append(model)
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Inference function using one-vs-all training strategy. Requires fit() function to be called prior to calling this
        Predictions are resolved by selecting the model output that is not 'other'
        If no model predicts the expected class, or if multiple classes are predicted, the class furthest from the decision boundary is selected

        Args:
            x (np.ndarray): Features of samples to predict on. Expected shape is (n_samples, n_features)

        Returns:
            np.ndarray: 1D array containing predictions. Shape is (n_samples, )
        """
        predictions = [model.predict(x) for model in self._models]
        prediction_scores = [model.decision_function(x) for model in self._models]

        resolved_predictions = []
        for i in range(len(x)):
            predictions_for_index = [predictions[j][i] for j in range(len(self._models)) if predictions[j][i] != 'other']
            # case where no class is predicted or multiple classes are predicted
            if len(predictions_for_index) != 1:
                # break tie using decision-function
                # documentation suggests that positive values indicate score for last class in model.classes_
                # https://scikit-learn.org/stable/glossary.html#term-decision_function
                decision_boundary_vals_for_index = [
                    (prediction_scores[j][i] if model.classes_[-1] != 'other' else prediction_scores[j][i] * -1)
                    for j, model in enumerate(self._models)
                ]
                best_val = np.argmax(decision_boundary_vals_for_index)
                resolved_class = next((class_label for class_label in self._models[best_val].classes_ if class_label != 'other'))
            else:
                resolved_class = predictions_for_index[0]

            resolved_predictions.append(resolved_class)

        return np.array(resolved_predictions)

# %%
ovo_classifier = OVOClassifier().fit(train_features, train_labels['class'])
y_pred_ovo = ovo_classifier.predict(test_features)
pretty_print_results(test_labels['class'], y_pred_ovo, 'OVO SVM')

# %%
ova_classifier = OVAClassifier().fit(train_features, train_labels['class'])
y_pred_ova = ova_classifier.predict(test_features)
pretty_print_results(test_labels['class'], y_pred_ova, 'OVA SVM')

# %% [markdown]
# Using the OVO approach with a linear SVM to classify the iris dataset gives a perfect classification for the test dataset considered here. The OVA approach produces a poorer classifier, compared to OVO, with 6 misclassifications between Iris-versicolor and Iris-virginica. Both classifiers are able to classify all the Iris-setosa in the test dataset.

# %% [markdown]
# To use the decision tree classifier, we use the DecisionTreeClassifier class from scikit-learn. Decision trees can easily support multiclass classification by using a one-vs-rest approach to split nodes. At each node, the problem can be posed as a binary classification problem to separate out the class that has the purest classification (or lowest gini). The decision tree may have multiple leaf nodes for each class, but can solve multi-class problems using just one tree as opposed to the multiple models required by approaches like SVM.

# %%
decision_tree = DecisionTreeClassifier(random_state=random_seed).fit(train_features, train_labels['class'])
y_pred_dtc = decision_tree.predict(test_features)
pretty_print_results(test_labels['class'], y_pred_dtc, 'Decision Tree Classifier')

# %% [markdown]
# The results of classifying the iris dataset using a decision tree with default parameters yields a performance that is better than that of OVA SVM, but lower than that of the OVO SVM. Overall, there are three misclassification between Iris-versicolor and Iris-virginica using this approach and it can be a good approach when OVO SVM is too slow for a large number of classes.


