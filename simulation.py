import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("white")

n = 5
kMact = np.random.exponential(size=(n, n))
kMdeg = np.random.exponential(size=(n, n))
kact = np.random.exponential(size=(n, n))
kdeg = np.random.exponential(size=(n, n))

kact0 = np.random.exponential(size=n)
kdeg0 = np.random.exponential(size=n)

epsilon = 1E-2
n_epsilon = 500
n_simulations = 10
n_interventions = 10

df = pd.DataFrame()

for k in range(n_interventions):
    if k == 0:
        ci = 1
    elif k <= n:
        ci = np.ones(n)
        ci[k-1] = 3
    else:
        ci = 1 + np.random.exponential(size=n)

    for j in range(n_simulations):
        c0 = np.random.exponential(size=n)
        c = c0
        for i in np.arange(0, n_epsilon):
            c += (-c*kdeg0 + kact0 + np.sum(kact/(1+kMact/(c/ci)), axis=1) - c*np.sum(kdeg/(1+kMact/(c/ci)), axis=1))*epsilon
            c = np.maximum(c, 1E-4)
        df = df.append(pd.DataFrame({"intervention": k, 'entity': np.arange(n), 'concentration': c, 'simulation_id': j}))


f, ax = plt.subplots(figsize=(7, 7))
ax.set(yscale="log")

sns.tsplot(data=df, time="intervention", unit="simulation_id",
           condition="entity", value="concentration", err_style="unit_traces")
plt.show()



