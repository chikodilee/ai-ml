# %%
import seaborn as sns
import matplotlib.pyplot as plt

# %%
plt.figure()
a = sns.lineplot(x=[10, 20, 30, 40, 50, 60, 70, 80, 90], y=[0.01, 0.09, 0.36, 0.64, 1.0, 0.49, 0.25, 0.09, 0.01], label = 'Presumably Fast')
b = sns.lineplot(x=[10, 20, 30, 40, 50, 60, 70, 80, 90], y=[0.1,0.3,0.6,0.8,1.0,0.7,0.5,0.3,0.1], label = "Fast")
a.fill_between([10, 20, 30, 40, 50, 60, 70, 80, 90], [0.01, 0.09, 0.36, 0.64, 1.0, 0.49, 0.25, 0.09, 0.01], alpha=0.5)
plt.xlabel('Speed (rev/s)')
plt.ylabel('Membership Value')
plt.legend()
plt.show()

a = sns.lineplot(x=list(map(lambda x: x**2, [10, 20, 30, 40, 50, 60, 70, 80, 90])), y=[0.1,0.3,0.6,0.8,1.0,0.7,0.5,0.3,0.1])
a.fill_between(x=list(map(lambda x: x**2, [10, 20, 30, 40, 50, 60, 70, 80, 90])), y1=[0.1,0.3,0.6,0.8,1.0,0.7,0.5,0.3,0.1], alpha=0.5)
plt.xlabel('Power')
plt.ylabel('Membership Value')
plt.show()



