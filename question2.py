# %% [markdown]
# We need to train a MLP network for obtaining the output of the following two-to-one mapping function.

# %%
import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

def mapping_function(x1, x2):
    """Mapping function that the MLP will try to learn

    Args:
        x1 (np.ndarray): 1D array of first feature of shape (n_samples, )
        x2 (np.ndarray): 1D array of second feature of shape (n_samples, )

    Returns:
        np.ndarray: 1D array of response variable of shape (n_samples, )
    """
    return np.sin(2 * math.pi * x1) * np.cos(0.5 * math.pi * x2) * np.exp( -1 * (x2 ** 2))

# %% [markdown]
# Set up two sets of data, one for network training and the other for testing (70% for training and 30% for testing). The total number of input-output data is 500 and is obtained by randomly varying the input variables (x1, x2) within the interval [-1,1] by [-4 4].

# %%
np.random.seed(RANDOM_SEED)
x1 = np.clip(np.random.uniform(low=-1.001, high=1.001, size=500), -1.0, 1.0)
x2 = np.clip(np.random.uniform(low=-4.001, high=4.001, size=500), -4.0, 4.0)

y = mapping_function(x1, x2)
x = np.moveaxis([x1, x2], 0, -1)

train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.7, random_state=RANDOM_SEED)

# %% [markdown]
# First, fix the number of hidden neurons to 4 (double of the number of input nodes) and analyze the performance of the obtained network (training and testing output errors). Use one hidden layer for the exercise.

# %%
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

def check_performance(y_true, y_pred):
    """Function to compare y_true to y_pred

    Args:
        y_true (np.ndarray): 1D array of GT values of shape (n_samples)
        y_pred (np.ndarray): 1D array of predicted values of shape (n_samples, )

    Returns:
        mse, mae, rmse (float, float, float): mean squared error, mean absolute error, root mean squared error
    """
    # we're only really looking at MSE, but calculating other metrics for understanding them
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    return mse, mae, rmse


mlp = MLPRegressor(hidden_layer_sizes=[4,], random_state=RANDOM_SEED).fit(train_x, train_y)
mpl_pred = mlp.predict(test_x)

mse, _, _ = check_performance(test_y, mpl_pred)
print('-'* 50)
print('Performance when hidden neurons = 4')
print(f'MSE: {mse}')
print('-'* 50)

# %% [markdown]
# Analyze the performance of the network with more and then with fewer hidden nodes (2, 6, 8, 12, 20). Find the best number of hidden neurons leading to the least training and testing network error and discuss.

# %%
hidden_neuron_iterations = [2, 4, 6, 8, 12, 20]

test_mse = []
train_mse = []

test_mse_for_random_seed = []
train_mse_for_random_seed = []

for random_state in [0, 1, 2, 3, 4, 5, RANDOM_SEED, 50, 100]:
    iter_train_mse = []
    iter_test_mse = []
    for hidden_neuron_size in hidden_neuron_iterations:
        mlp = MLPRegressor(hidden_layer_sizes=[hidden_neuron_size, ], random_state=random_state).fit(train_x, train_y)
        test_predictions = mlp.predict(test_x)
        train_predictions = mlp.predict(train_x)

        test_mse_val, _, _ = check_performance(test_y, test_predictions)
        train_mse_val, _, _ = check_performance(train_y, train_predictions)
        iter_test_mse.append(test_mse_val)
        iter_train_mse.append(train_mse_val)
    
    if random_state == RANDOM_SEED:
        test_mse_for_random_seed = iter_test_mse
        train_mse_for_random_seed = iter_train_mse
    
    test_mse.append(iter_test_mse)
    train_mse.append(iter_train_mse)

train_mse = np.array(train_mse).mean(0)
test_mse = np.array(test_mse).mean(0) 

# %%
print('-'*50)
print(f'MSE values for random_state=42')
print(
    pd.DataFrame(
        np.array([hidden_neuron_iterations, iter_train_mse, iter_test_mse]).T,
        columns=['Hidden neurons', 'Train MSE', 'Test MSE']
    )
)

# %%
sns.lineplot(x=hidden_neuron_iterations, y=train_mse_for_random_seed, marker='o', label='Train MSE')
sns.lineplot(x=hidden_neuron_iterations, y=test_mse_for_random_seed, marker='o', label='Test MSE')
plt.legend()
plt.xlabel('Number of hidden neurons')
plt.ylabel('MSE')
plt.xticks(hidden_neuron_iterations)
plt.title("Train and test MSE for random_state=42")
plt.show()

# %%
print('-'*50)
print(f'MSE values after averaging')
print(
    pd.DataFrame(
        np.array([hidden_neuron_iterations, train_mse, test_mse]).T,
        columns=['Hidden neurons', 'Averaged train MSE', 'Averaged test MSE']
    )
)

# %%
sns.lineplot(x=hidden_neuron_iterations, y=train_mse, marker='o', label='Train MSE')
sns.lineplot(x=hidden_neuron_iterations, y=test_mse, marker='o', label='Test MSE')
plt.legend()
plt.xlabel('Number of hidden neurons')
plt.ylabel('MSE')
plt.xticks(hidden_neuron_iterations)
plt.title("Train and test MSE averaged over different random states")
plt.show()

