import matplotlib
matplotlib.use('Agg')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.rcParams.update({'font.size': 14})

os.chdir("C:/Users/...Experiment 2/Analyses")

data = pd.read_csv("Behavioural_Data.csv")
data = data.loc[(data["trial"] != 1) & (data["trial"] != 10)].reset_index()

label_list = ["low noise", "medium noise", "high noise"]

data["error"] = np.nan
data["learn"] = np.nan

for trial in range(len(data)):
    
    if (data["trial"][trial] != 2) and (data["crab"][trial-1] != data["cage"][trial-1]):
        
        error = data["crab"][trial-1] - data["cage"][trial-1]
        data["error"][trial] = abs(error)
        update = data["cage"][trial] - data["cage"][trial-1]
        data["learn"][trial] = update / error
            
data = data.dropna()

fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (12.8, 14.4))

mat = np.zeros([53, 3, 7])

for part in range(53):
    for sigma in range(3):
        for trial in range(7):
            
            frame = data.loc[(data["part"] == part+1) & (data["sigma"] == sigma+1) & (data["trial"] == trial+3)]
            mat[part, sigma, trial] = np.median(frame["learn"])

mean_mat = np.zeros([3, 7])
sd_mat = np.zeros([3, 7])

for sigma in range(3):
    for trial in range(7):
        
        mean_mat[sigma, trial] = np.mean(mat[:, sigma, trial])
        sd_mat[sigma, trial] = stats.sem(mat[:, sigma, trial], nan_policy = "omit")

for sigma in range(3):
    axs[0, 0].errorbar([2, 3, 4, 5, 6, 7, 8], mean_mat[sigma, :], yerr = sd_mat[sigma, :], label = label_list[sigma])
    
axs[0, 0].set_xlabel("trials")
axs[0, 0].set_ylim(0.18, 0.78)
axs[0, 0].set_ylabel("learning rates")
axs[0, 0].legend()
axs[0, 0].set_title("Entire task")
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)

frame = pd.DataFrame({
    "low": mat[:, 0, 0],
    "medium": mat[:, 1, 0],
    "high": mat[:, 2, 0]
})

for index in range(53):
    axs[0, 1].plot([0, 1, 2], [mat[index, 0, 0], mat[index, 1, 0], mat[index, 2, 0]], color = "gray", alpha = 0.25)
    axs[0, 1].scatter([0, 1, 2], [mat[index, 0, 0], mat[index, 1, 0], mat[index, 2, 0]], color = "gray", alpha = 0.25)

sns.pointplot(data = frame, join = True, errorbar = "se", capsize = 0.2, color = "black", ax = axs[0, 1])

axs[0, 1].set_xticks([0, 1, 2], label_list)
axs[0, 1].set_ylim([-0.08, 1.08])
axs[0, 1].set_ylabel("learning rates")
axs[0, 1].set_title("Trial 2 entire task")
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].spines['right'].set_visible(False)

plt.tight_layout()

mat = np.zeros([53, 3, 7])

for part in range(53):
    for sigma in range(3):
        for trial in range(7):
            
            frame = data.loc[(data["part"] == part+1) & (data["sigma"] == sigma+1) & (data["trial"] == trial+3) & (data["run"] == 1)]
            mat[part, sigma, trial] = np.median(frame["learn"])

mean_mat = np.zeros([3, 7])
sd_mat = np.zeros([3, 7])

for sigma in range(3):
    for trial in range(7):
        
        mean_mat[sigma, trial] = np.mean(mat[:, sigma, trial])
        sd_mat[sigma, trial] = stats.sem(mat[:, sigma, trial], nan_policy = "omit")

for sigma in range(3):
    axs[1, 0].errorbar([2, 3, 4, 5, 6, 7, 8], mean_mat[sigma, :], yerr = sd_mat[sigma, :], label = label_list[sigma])
    
axs[1, 0].set_xlabel("trials")
axs[1, 0].set_ylim(0.18, 0.78)
axs[1, 0].set_ylabel("learning rates")
axs[1, 0].set_title("First quarter of task")
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].spines['right'].set_visible(False)

mat = np.zeros([53, 3, 7])

for part in range(53):
    for sigma in range(3):
        for trial in range(7):
            
            frame = data.loc[(data["part"] == part+1) & (data["sigma"] == sigma+1) & (data["trial"] == trial+3) & (data["run"] == 2)]
            mat[part, sigma, trial] = np.median(frame["learn"])

mean_mat = np.zeros([3, 7])
sd_mat = np.zeros([3, 7])

for sigma in range(3):
    for trial in range(7):
        
        mean_mat[sigma, trial] = np.mean(mat[:, sigma, trial])
        sd_mat[sigma, trial] = stats.sem(mat[:, sigma, trial], nan_policy = "omit")

for sigma in range(3):
    axs[1, 1].errorbar([2, 3, 4, 5, 6, 7, 8], mean_mat[sigma, :], yerr = sd_mat[sigma, :], label = label_list[sigma])
    
axs[1, 1].set_xlabel("trials")
axs[1, 1].set_ylim(0.18, 0.78)
axs[1, 1].set_ylabel("learning rates")
axs[1, 1].set_title("Second quarter of task")
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)

mat = np.zeros([53, 3, 7])

for part in range(53):
    for sigma in range(3):
        for trial in range(7):
            
            frame = data.loc[(data["part"] == part+1) & (data["sigma"] == sigma+1) & (data["trial"] == trial+3) & (data["run"] == 3)]
            mat[part, sigma, trial] = np.median(frame["learn"])

mean_mat = np.zeros([3, 7])
sd_mat = np.zeros([3, 7])

for sigma in range(3):
    for trial in range(7):
        
        mean_mat[sigma, trial] = np.mean(mat[:, sigma, trial])
        sd_mat[sigma, trial] = stats.sem(mat[:, sigma, trial], nan_policy = "omit")

for sigma in range(3):
    axs[2, 0].errorbar([2, 3, 4, 5, 6, 7, 8], mean_mat[sigma, :], yerr = sd_mat[sigma, :], label = label_list[sigma])
    
axs[2, 0].set_xlabel("trials")
axs[2, 0].set_ylim(0.18, 0.78)
axs[2, 0].set_ylabel("learning rates")
axs[2, 0].set_title("Third quarter of task")
axs[2, 0].spines['top'].set_visible(False)
axs[2, 0].spines['right'].set_visible(False)

mat = np.zeros([53, 3, 7])

for part in range(53):
    for sigma in range(3):
        for trial in range(7):
            
            frame = data.loc[(data["part"] == part+1) & (data["sigma"] == sigma+1) & (data["trial"] == trial+3) & (data["run"] == 4)]
            mat[part, sigma, trial] = np.median(frame["learn"])

mean_mat = np.zeros([3, 7])
sd_mat = np.zeros([3, 7])

for sigma in range(3):
    for trial in range(7):
        
        mean_mat[sigma, trial] = np.mean(mat[:, sigma, trial])
        sd_mat[sigma, trial] = stats.sem(mat[:, sigma, trial], nan_policy = "omit")

for sigma in range(3):
    axs[2, 1].errorbar([2, 3, 4, 5, 6, 7, 8], mean_mat[sigma, :], yerr = sd_mat[sigma, :], label = label_list[sigma])
    
axs[2, 1].set_xlabel("trials")
axs[2, 1].set_ylim(0.18, 0.78)
axs[2, 1].set_ylabel("learning rates")
axs[2, 1].set_title("Fourth quarter of task")
axs[2, 1].spines['top'].set_visible(False)
axs[2 ,1].spines['right'].set_visible(False)

plt.tight_layout()

plt.gcf().text(0.015, 0.975, "A", fontsize = 16, fontweight = "bold")
plt.gcf().text(0.015, 0.65, "B", fontsize = 16, fontweight = "bold")

fig.savefig("Figure_1")

import statsmodels.formula.api as smf

data_dict = {
    "part": [],
    "env": [],
    "trial": [],
    "learn": []
}

for part in range(53):
    for env in range(3):
        for trial in range(7):
                
            frame = data.loc[(data["part"] == part+1) & (data["sigma"] == env+1) & (data["trial"] == trial+3)]
            
            data_dict["part"].append(part+1)
            data_dict["env"].append(env+1)
            data_dict["trial"].append(trial+1-5)
            data_dict["learn"].append(np.median(frame["learn"]))

frame = pd.DataFrame(data = data_dict)

frame["trial"] = frame["trial"].astype(float)

md = smf.mixedlm("learn ~ env*trial", frame, groups = frame["part"], re_formula = "~env*trial")
mdf = md.fit(method = ["lbfgs"])
print(mdf.summary())

from statsmodels.stats.anova import AnovaRM

data_dict = {
    "part": [],
    "run": [],
    "env": [],
    "learn": []
}

for part in range(53):
    for run in range(4):
        for env in range(3):
                
            frame = data.loc[(data["part"] == part+1) & (data["run"] == run+1) & (data["sigma"] == env+1) & (data["trial"] == 3)]
            
            data_dict["part"].append(part+1)
            data_dict["run"].append(run+1)
            data_dict["env"].append(env+1)
            data_dict["learn"].append(np.median(frame["learn"]))

frame = pd.DataFrame(data = data_dict)

print(AnovaRM(data = frame, depvar = "learn", subject = "part", within = ["run", "env"]).fit())
