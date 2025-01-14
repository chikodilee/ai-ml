# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %%
def general_fn(l, alpha, n):
    def fn(x):
        return np.exp(-1 * l * (np.abs(x - alpha) ** n))
    return fn

def cold_fn(x):
    return general_fn(l=0.01, n=2.0, alpha=0)(x) * (x >= 0).astype(np.float32) + 1.0 * (x < 0).astype(np.float32)

def comf_fn(x):
    return general_fn(l=0.01, n=3.5, alpha=21.0)(x)

def hot_fn(x):
    return general_fn(l=0.05, n=2.0, alpha=30.0)(x) * (x <= 30).astype(np.float32) + 1.0 * (x > 30).astype(np.float32)

temps = np.linspace(-5, 35, 1000)
plt.figure(figsize=(8, 5))
a=sns.lineplot(x=temps, y=cold_fn(temps))
b=sns.lineplot(x=temps, y=hot_fn(temps))
c=sns.lineplot(x=temps, y=comf_fn(temps))

a.fill_between(temps, cold_fn(temps), alpha=0.5)
b.fill_between(temps, hot_fn(temps), alpha=0.5)
c.fill_between(temps, comf_fn(temps), alpha=0.5)

plt.ylabel('Membership value')
plt.xlabel('Temperature (C)')
plt.legend(['cold', 'hot', 'comfortable'], title='state', loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=False, ncol=5)
plt.grid(True)
plt.xticks([-5, 0, 21, 30, 35])
plt.xlim(-5, 35)
# plt.title('Membership values of differnt temperature states')
plt.show()
