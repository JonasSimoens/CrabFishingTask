import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("C:/Users/jlsimoen/OneDrive - UGent/Documents/Research/Neuroimaging/Analyses")

plt.rcParams.update({'font.size': 14})

label_list = ["low noise", "medium noise", "high noise"]

trace = pd.read_csv("Hybrid_Model_Trace.csv")

plt.ioff()

low_sigma_alfa = sns.kdeplot(trace["low_sigma_learn_mu"]).get_lines()[0].get_ydata()
plt.close()
medium_sigma_alfa = sns.kdeplot(trace["medium_sigma_learn_mu"]).get_lines()[0].get_ydata()
plt.close()
high_sigma_alfa = sns.kdeplot(trace["high_sigma_learn_mu"]).get_lines()[0].get_ydata()
plt.close()
max_alfa = np.max(np.concatenate((low_sigma_alfa, medium_sigma_alfa, high_sigma_alfa)))


low_sigma_eta = sns.kdeplot(trace["low_sigma_decay_mu"]).get_lines()[0].get_ydata()
plt.close()
medium_sigma_eta = sns.kdeplot(trace["medium_sigma_decay_mu"]).get_lines()[0].get_ydata()
plt.close()
high_sigma_eta = sns.kdeplot(trace["high_sigma_decay_mu"]).get_lines()[0].get_ydata()
plt.close()
max_eta = np.max(np.concatenate((low_sigma_eta, medium_sigma_eta, high_sigma_eta)))

plt.ion()

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (12.8, 4.8))

sns.kdeplot(y = trace["low_sigma_learn_mu"], fill = True, ax = axs[0])
sns.kdeplot(y = trace["medium_sigma_learn_mu"], fill = True, ax = axs[0])
sns.kdeplot(y = trace["high_sigma_learn_mu"], fill = True, ax = axs[0])

low_sigma_learn = np.zeros(53)
medium_sigma_learn = np.zeros(53)
high_sigma_learn = np.zeros(53)

for part in range(53):
    
    low_sigma_learn[part] = np.mean(trace["low_sigma_learn_part_{}".format(part+1)])
    medium_sigma_learn[part] = np.mean(trace["medium_sigma_learn_part_{}".format(part+1)])
    high_sigma_learn[part] = np.mean(trace["high_sigma_learn_part_{}".format(part+1)])
    
axs[0].scatter(np.random.uniform(max_alfa * 1.2, max_alfa * 1.25, size = 53), low_sigma_learn, color = "#c6dcec", edgecolors = "#1f77b4")
axs[0].scatter(np.random.uniform(max_alfa * 1.45, max_alfa * 1.5, size = 53), medium_sigma_learn, color = "#ffdec2", edgecolors = "#ff7f0e")
axs[0].scatter(np.random.uniform(max_alfa * 1.65, max_alfa * 1.7, size = 53), high_sigma_learn, color = "#cae7ca", edgecolors = "#2ca02c")

axs[0].set_xlim(0, max_alfa * 1.8)
axs[0].get_xaxis().set_visible(False)
axs[0].set_ylabel("initial learning rates")
axs[0].set_title("Initial learning rate estimation")

axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['bottom'].set_visible(False)

sns.kdeplot(y = trace["low_sigma_decay_mu"], fill = True, ax = axs[1], label = label_list[0])
sns.kdeplot(y = trace["medium_sigma_decay_mu"], fill = True, ax = axs[1], label = label_list[1])
sns.kdeplot(y = trace["high_sigma_decay_mu"], fill = True, ax = axs[1], label = label_list[2])

low_sigma_decay = np.zeros(53)
medium_sigma_decay = np.zeros(53)
high_sigma_decay = np.zeros(53)

for part in range(53):
    
    low_sigma_decay[part] = np.mean(trace["low_sigma_decay_part_{}".format(part+1)])
    medium_sigma_decay[part] = np.mean(trace["medium_sigma_decay_part_{}".format(part+1)])
    high_sigma_decay[part] = np.mean(trace["high_sigma_decay_part_{}".format(part+1)])
    
axs[1].scatter(np.random.uniform(max_eta * 1.2, max_eta * 1.25, size = 53), low_sigma_decay, color = "#c6dcec", edgecolors = "#1f77b4")
axs[1].scatter(np.random.uniform(max_eta * 1.45, max_eta * 1.5, size = 53), medium_sigma_decay, color = "#ffdec2", edgecolors = "#ff7f0e")
axs[1].scatter(np.random.uniform(max_eta * 1.65, max_eta * 1.7, size = 53), high_sigma_decay, color = "#cae7ca", edgecolors = "#2ca02c")

axs[1].set_xlim(0, max_eta * 1.8)
axs[1].get_xaxis().set_visible(False)
axs[1].set_ylabel("decay rates")
axs[1].set_title("Decay rate estimation")
axs[1].legend()

axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].spines['bottom'].set_visible(False)

plt.tight_layout()

fig.savefig("Figure_2")

count = trace["low_sigma_learn_mu"] > trace["high_sigma_learn_mu"]
print("P(low sigma > high sigma) = {}".format(sum(count)/len(count)))

count = trace["low_sigma_learn_mu"] > trace["medium_sigma_learn_mu"]
print("P(low sigma > medium sigma) = {}".format(sum(count)/len(count)))

count = trace["medium_sigma_learn_mu"] > trace["high_sigma_learn_mu"]
print("P(medium sigma > high sigma) = {}".format(sum(count)/len(count)))
