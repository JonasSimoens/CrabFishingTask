import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

os.chdir("C:/Users/.../Experiment 2/Analyses")

#%%

data = pd.DataFrame()

for alfa_loop in range(4):
    for eta_loop in range(4):
        
            alfa = np.round((alfa_loop + 1) * 0.2, 1)
            eta = np.round((eta_loop + 1) * 0.2, 1)
            
            frame = pd.read_csv("env_1_alfa_{}_eta_{}.csv".format(alfa, eta))
            data_frame = [data, frame]
            data = pd.concat(data_frame)

#%%

plt.scatter(data["true_learn"], data["est_learn"])
plt.tight_layout()

print(stats.pearsonr(data["true_learn"], data["est_learn"]))

#%%

plt.scatter(data["true_decay"], data["est_decay"])
plt.tight_layout()

print(stats.pearsonr(data["true_decay"], data["est_decay"]))

#%%

data = pd.DataFrame()

for alfa_loop in range(4):
    for eta_loop in range(4):
        
            alfa = np.round((alfa_loop + 1) * 0.2, 1)
            eta = np.round((eta_loop + 1) * 0.2, 1)
            
            frame = pd.read_csv("env_2_alfa_{}_eta_{}.csv".format(alfa, eta))
            data_frame = [data, frame]
            data = pd.concat(data_frame)

#%%

plt.scatter(data["true_learn"], data["est_learn"])
plt.tight_layout()

print(stats.pearsonr(data["true_learn"], data["est_learn"]))

#%%

plt.scatter(data["true_decay"], data["est_decay"])
plt.tight_layout()

print(stats.pearsonr(data["true_decay"], data["est_decay"]))

#%%

data = pd.DataFrame()

for alfa_loop in range(4):
    for eta_loop in range(4):
        
            alfa = np.round((alfa_loop + 1) * 0.2, 1)
            eta = np.round((eta_loop + 1) * 0.2, 1)
            
            frame = pd.read_csv("env_3_alfa_{}_eta_{}.csv".format(alfa, eta))
            data_frame = [data, frame]
            data = pd.concat(data_frame)

#%%

plt.scatter(data["true_learn"], data["est_learn"])
plt.tight_layout()

print(stats.pearsonr(data["true_learn"], data["est_learn"]))

#%%

plt.scatter(data["true_decay"], data["est_decay"])
plt.tight_layout()

print(stats.pearsonr(data["true_decay"], data["est_decay"]))
