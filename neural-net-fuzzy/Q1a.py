import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


def function1(x):
    return x * np.sin(6 * np.pi * x) * np.exp(-x**2)

def function2(x):
    return np.exp(-x**2) * np.arctan(x) * np.sin(4 * np.pi * x)

def modeling(xtraining, xtest, ytraining, ytest):
    model.fit(xtraining.reshape(-1,1), ytraining)
    ypredicttraining = model.predict(xtraining.reshape(-1,1))
    ypredicttest = model.predict(xtest.reshape(-1,1))
    trainingerror = np.mean((ypredicttraining - ytraining)**2)
    testerror = np.mean((ypredicttest - ytest)**2)

    return trainingerror, testerror

def plotting(samplesize, trainingerror, testerror):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(samplesize, trainingerror, label='Training Error')
    plt.plot(samplesize, testerror, label='Testing Error')
    plt.xlabel('Sample Size')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    plt.tight_layout()
    plt.show()


numpoints = 1000
samplesize = np.arange(20, 201, 1)

np.random.seed(70)

x1 = np.random.uniform(-1, 1, numpoints)
x2 = np.random.uniform(-2, 2, numpoints)

y1 = function1(x1)
y2 = function2(x2)

model = MLPRegressor(hidden_layer_sizes=(3,), activation='logistic', solver='adam', max_iter=1000)

func1trainingerror = []
func1testerror = []
func2trainingerror = []
func2testerror = []

for sample in samplesize:
    x1training, x1test, y1training, y1test = train_test_split(x1, y1, test_size=0.3)
    func1 = modeling(x1training, x1test, y1training, y1test)
    func1trainingerror.append(func1[0])
    func1testerror.append(func1[1])

    x2training, x2test, y2training, y2test = train_test_split(x2, y2, test_size=0.3)
    func2 = modeling(x2training, x2test, y2training, y2test)
    func2trainingerror.append(func2[0])
    func2testerror.append(func2[1])

plotting(samplesize, func1trainingerror, func1testerror)
plotting(samplesize, func2trainingerror, func2testerror)





    

