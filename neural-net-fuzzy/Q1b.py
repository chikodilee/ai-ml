import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt


def function1(x):
    return x * np.sin(6 * np.pi * x) * np.exp(-x**2)

def function2(x):
    return np.exp(-x**2) * np.arctan(x) * np.sin(4 * np.pi * x)

def modeling(hiddennode, xtraining, xtest, ytraining, ytest):
    model = MLPRegressor(hidden_layer_sizes=(hiddennode,), activation='logistic', solver='adam', max_iter=1000)
    model.fit(xtraining.reshape(-1,1), ytraining)
    ypredicttraining = model.predict(xtraining.reshape(-1,1))
    ypredicttest = model.predict(xtest.reshape(-1,1))
    trainingerror = np.mean((ypredicttraining - ytraining)**2)
    testerror = np.mean((ypredicttest - ytest)**2)

    return trainingerror, testerror

def plotting(hiddennodes, optimal_hidden_nodes, trainingerrordict, testerrordict):
    trainingerror = [trainingerrordict[hiddennode] for hiddennode in hiddennodes]
    testerror = [testerrordict[hiddennode] for hiddennode in hiddennodes]
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(hiddennodes, trainingerror, label='Training Error')
    plt.plot(hiddennodes, testerror, label='Testing Error')
    plt.xlabel('Number of Hidden Nodes')
    plt.ylabel('Mean squared Score')
    plt.axvline(x=optimal_hidden_nodes, color='r', linestyle='--', label=f'Optimal Nodes: {optimal_hidden_nodes}')
    plt.legend()

    plt.tight_layout()
    plt.show()


numpoints = 1000
samplesize = 100
hiddennodes = range(1, 101)

np.random.seed(70)

x1 = np.random.uniform(-1, 1, numpoints)
x2 = np.random.uniform(-2, 2, numpoints)

y1 = function1(x1)
y2 = function2(x2)

func1trainingerror = {}
func1testerror = {}
func1avgtrainingerror = {}
func1avgtesterror = {}

func2trainingerror = {}
func2testerror = {}
func2avgtrainingerror = {}
func2avgtesterror ={}

for i in range(samplesize):
    x1training, x1test, y1training, y1test = train_test_split(x1, y1, test_size=0.3)
    x2training, x2test, y2training, y2test = train_test_split(x2, y2, test_size=0.3)

    for hiddennode in hiddennodes:
        func1 = modeling(hiddennode, x1training, x1test, y1training, y1test)
        if hiddennode not in func1trainingerror:
            func1trainingerror[hiddennode] = [func1[0]]
            func1testerror[hiddennode] = [func1[1]]
        else:
            func1trainingerror[hiddennode].append(func1[0])
            func1testerror[hiddennode].append(func1[1])

        func2 = modeling(hiddennode, x2training, x2test, y2training, y2test)
        if hiddennode not in func2trainingerror:
            func2trainingerror[hiddennode] = [func2[0]]
            func2testerror[hiddennode] = [func2[1]]
        else:
            func2trainingerror[hiddennode].append(func2[0])
            func2testerror[hiddennode].append(func2[1])

for hiddennode in func1trainingerror.keys():
    func1avgtrainingerror[hiddennode] = np.mean(func1trainingerror[hiddennode])
    func1avgtesterror[hiddennode] = np.mean(func1testerror[hiddennode])
    func2avgtrainingerror[hiddennode] = np.mean(func2trainingerror[hiddennode])
    func2avgtesterror[hiddennode] = np.mean(func2testerror[hiddennode])

min_func1_test_error = min(func1avgtesterror.values())
func1optimal_hidden_nodes = next(key for key, value in func1avgtesterror.items() if value == min_func1_test_error)

min_func2_test_error = min(func2avgtesterror.values())
func2optimal_hidden_nodes = next(key for key, value in func2avgtesterror.items() if value == min_func2_test_error)

print("Optimal Number of Hidden Nodes for Function 1:", func1optimal_hidden_nodes)
print("Optimal Number of Hidden Nodes for Function 2:", func2optimal_hidden_nodes)

plotting(hiddennodes, func1optimal_hidden_nodes, func1avgtrainingerror, func1avgtesterror)
plotting(hiddennodes, func2optimal_hidden_nodes, func2avgtrainingerror, func2avgtesterror)
