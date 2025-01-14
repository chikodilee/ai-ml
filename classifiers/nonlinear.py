import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

data = pd.read_csv("./data/DataC.csv")

def digit3visual(digit3indices):
    numimages = 10
    randomimageindices = []

    for i in range(numimages):
        randomimageindices.append(np.random.choice(digit3indices))

    fig, axes = plt.subplots(1, numimages, figsize=(12, 4))

    for i, idx in enumerate(randomimageindices[:numimages]):
        image = np.array(data.iloc[idx][:784]).reshape(28, 28) 
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')

    plt.show()

def plot(data, algorithm, digit3indices, title, xlabel, ylabel):
    plt.figure(figsize=(8, 6))
    plt.scatter(algorithm[:, 0], algorithm[:, 1], c=data.iloc[digit3indices, -1], cmap='tab10', edgecolor='k', s=50)
    # plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def acc(X_train, X_test, y_train, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

def NaiveBayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lle = LocallyLinearEmbedding(n_neighbors=5, n_components=4)
    lletraining = lle.fit_transform(X_train)
    lletest = lle.transform(X_test)

    isomap = Isomap(n_neighbors=5, n_components=4)
    isomaptraining = isomap.fit_transform(X_train)
    isomaptest = isomap.transform(X_test)

    pca = PCA(n_components=4)
    pcatraining = pca.fit_transform(X_train)
    pcatest = pca.transform(X_test)

    lda = LDA(n_components=4)
    ldatraining = lda.fit_transform(X_train, y_train)
    ldatest = lda.transform(X_test)
    
    lleaccuracy = acc(lletraining, lletest, y_train, y_test)
    isomapaccuracy = acc(isomaptraining, isomaptest, y_train, y_test)
    pcaaccuracy = acc(pcatraining, pcatest, y_train, y_test)
    ldaaccuracy = acc(ldatraining, ldatest, y_train, y_test)

    return lleaccuracy, isomapaccuracy, pcaaccuracy, ldaaccuracy

# Extract images of digit '3'
digit3indices = []
for i in range(len(data)):
    image = np.array(data.iloc[i][:784]).reshape(28, 28)
    if data.iloc[i][-1] == 3:  # Check if the image represents digit '3'
        digit3indices.append(i)

# Visualize the images of digit '3'
digit3visual(digit3indices)

# LLE
lle = LocallyLinearEmbedding(n_neighbors=5, n_components=2)
lle_representation = lle.fit_transform(data.iloc[digit3indices, :-1])
plot(data, lle_representation, digit3indices, 'Locally Linear Embedding (LLE)', 'First Component', 'Second Component')

# Isomap
isomap = Isomap(n_neighbors=5, n_components=2)
isomap_representation = isomap.fit_transform(data.iloc[digit3indices, :-1])
plot(data, isomap_representation, digit3indices, 'Isomap', 'First Component', 'Second Component')

# Naive Bayes
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

numiterations = 60
numiterations2 = 80
numiterations3 = 100

accuracies = []
lleaccuracies = []
lleaccuracies2 = []
lleaccuracies3 = []
isomapaccuracies = []
isomapaccuracies2 = []
isomapaccuracies3 = []
pcaaccuracies = []
ldaaccuracies = []

for i in range(numiterations3):
    accuracies.append(NaiveBayes(X,y))
    if i in range(numiterations):
        lleaccuracies.append(accuracies[i][0])
        isomapaccuracies.append(accuracies[i][1])
    if i in range(numiterations2):
        lleaccuracies2.append(accuracies[i][0])
        isomapaccuracies2.append(accuracies[i][1])

    lleaccuracies3.append(accuracies[i][0])
    isomapaccuracies3.append(accuracies[i][1])
    pcaaccuracies.append(accuracies[i][2])
    ldaaccuracies.append(accuracies[i][3])

lleavgaccuracy = np.mean(lleaccuracies)
lleavgaccuracy2 = np.mean(lleaccuracies2)
lleavgaccuracy3 = np.mean(lleaccuracies3)
isomapavgaccuracy = np.mean(isomapaccuracies)
isomapavgaccuracy2 = np.mean(isomapaccuracies2)
isomapavgaccuracy3 = np.mean(isomapaccuracies3)
pcaavgaccuracy = np.mean(pcaaccuracies)
ldaavgaccuracy = np.mean(ldaaccuracies)

print("LLE Accuracy:", round(lleavgaccuracy*100, 2),"%", ",", round(lleavgaccuracy2*100, 2),"%", ",", round(lleavgaccuracy3*100, 2),"%")
print("Isomap Accuracy:", round(isomapavgaccuracy*100, 2),"%", ",", round(isomapavgaccuracy2*100, 2),"%", ",", round(isomapavgaccuracy3*100, 2),"%")
print("PCA Accuracy:", round(pcaavgaccuracy*100, 2),"%")
print("LDA Accuracy:", round(ldaavgaccuracy*100, 2),"%")
