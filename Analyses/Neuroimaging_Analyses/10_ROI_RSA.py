import os
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

os.chdir("C:/Users/.../Experiment 2/Analyses")

part_list = np.concatenate([range(1, 12), range(13, 20), range(21, 26), range(27, 52), range(53, 54)])

geometric_RDM = np.array([[0, 1, 2, 3, 2, 1],
                          [1, 0, 1, 2, 3, 2],
                          [2, 1, 0, 1, 2, 3],
                          [3, 2, 1, 0, 1, 2],
                          [2, 3, 2, 1, 0, 1],
                          [1, 2, 3, 2, 1, 0]])

geometric_RDV = geometric_RDM.reshape((-1,))

loc_pre_list = []

for part in part_list:
    
    neural_data = pd.read_csv("beta_values/beta_values_part_{}.csv".format(part))
    
    neural_data = neural_data.dropna()
    
    neural_RDM = []
                    
    for row in range(6):
        
        data_row = neural_data.iloc[:, row]
        
        for col in range(6):
            
            data_col = neural_data.iloc[:, col+6]
            
            neural_RDM.append(1 - st.pearsonr(data_row, data_col)[0])
        
    loc_pre_list.append(st.kendalltau(geometric_RDV, neural_RDM)[0])
    
loc_post_list = []
    
for part in part_list:
    
    neural_data = pd.read_csv("beta_values/beta_values_part_{}.csv".format(part))
    
    neural_data = neural_data.dropna()
    
    neural_RDM = []
                    
    for row in range(6):
        
        data_row = neural_data.iloc[:, row+12]
        
        for col in range(6):
            
            data_col = neural_data.iloc[:, col+18]
            
            neural_RDM.append(1 - st.pearsonr(data_row, data_col)[0])
    
    loc_post_list.append(st.kendalltau(geometric_RDV, neural_RDM)[0])

behavioural_data = pd.read_csv("Behavioural_Data.csv")

learn_pre_list = []

for part in part_list:
    
    behavioural_RDM = []
    
    for row in range(6):
        
        data_row = behavioural_data.loc[(behavioural_data["part"] == part) & (behavioural_data["angle"] == row+1)].reset_index()
        data_row = data_row["sigma"][0]
        
        for col in range(6):
            
            data_col = behavioural_data.loc[(behavioural_data["part"] == part) & (behavioural_data["angle"] == col+1)].reset_index()
            data_col = data_col["sigma"][0]
            
            behavioural_RDM.append(abs(data_row - data_col))

    behavioural_RDV = behavioural_RDM
    
    neural_data = pd.read_csv("beta_values/beta_values_part_{}.csv".format(part))
    
    neural_data = neural_data.dropna()
    
    neural_RDM = []
                    
    for row in range(6):
        
        data_row = neural_data.iloc[:, row]
        
        for col in range(6):
            
            data_col = neural_data.iloc[:, col+6]
            
            neural_RDM.append(1 - st.pearsonr(data_row, data_col)[0])
    
    learn_pre_list.append(st.kendalltau(behavioural_RDV, neural_RDM)[0])
    
learn_post_list = []

for part in part_list:
    
    behavioural_RDM = []
    
    for row in range(6):
        
        data_row = behavioural_data.loc[(behavioural_data["part"] == part) & (behavioural_data["angle"] == row+1)].reset_index()
        data_row = data_row["sigma"][0]
        
        for col in range(6):
            
            data_col = behavioural_data.loc[(behavioural_data["part"] == part) & (behavioural_data["angle"] == col+1)].reset_index()
            data_col = data_col["sigma"][0]
            
            behavioural_RDM.append(abs(data_row - data_col))

    behavioural_RDV = behavioural_RDM
    
    neural_data = pd.read_csv("beta_values/beta_values_part_{}.csv".format(part))
    
    neural_data = neural_data.dropna()
    
    neural_RDM = []
                    
    for row in range(6):
        
        data_row = neural_data.iloc[:, row+12]
        
        for col in range(6):
            
            data_col = neural_data.iloc[:, col+18]
            
            neural_RDM.append(1 - st.pearsonr(data_row, data_col)[0])
    
    learn_post_list.append(st.kendalltau(behavioural_RDV, neural_RDM)[0])

plt.rcParams.update({'font.size': 14})
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

label_list = ["spatial\nsimilarity\nfirst half", "spatial\nsimilarity\nsecond half", "semantic\nsimilarity\nfirst half", "semantic\nsimilarity\nsecond half"]

frame = pd.DataFrame({
    "loc_pre": loc_pre_list,
    "loc_post": loc_post_list,
    "learn_pre": learn_pre_list,
    "learn_post": learn_post_list
})

for index in range(49):
    plt.plot([0, 1], [loc_pre_list[index], loc_post_list[index]], color = "gray", alpha = 0.25)
    plt.scatter([0, 1], [loc_pre_list[index], loc_post_list[index]], color = "gray", alpha = 0.25)
    plt.plot([2, 3], [learn_pre_list[index], learn_post_list[index]], color = "gray", alpha = 0.25)
    plt.scatter([2, 3], [learn_pre_list[index], learn_post_list[index]], color = "gray", alpha = 0.25)

x_positions = [0, 1, 2, 3]
means = np.mean(frame, axis = 0)

for index in range(3):
    if index != 1:
        plt.plot(x_positions[index:index+2], means[index:index+2], color = "black", linewidth = 3)

sns.pointplot(data = frame, join = False, errorbar = "se", capsize = 0.2, color = "black")

plt.xticks([0, 1, 2, 3], label_list)
plt.ylabel("Kendall's tau-values")
plt.title("Central OFC")

plt.tight_layout()

plt.savefig("Figure_3")

data_dict = {
    "part": [],
    "cond": [],
    "run": [],
    "corr": []
}

for part in range(49):
    
    data_dict["part"].append(part+1)
    data_dict["cond"].append(1)
    data_dict["run"].append(1)
    data_dict["corr"].append(loc_pre_list[part])
    
    data_dict["part"].append(part+1)
    data_dict["cond"].append(1)
    data_dict["run"].append(2)
    data_dict["corr"].append(loc_post_list[part])
    
    data_dict["part"].append(part+1)
    data_dict["cond"].append(2)
    data_dict["run"].append(1)
    data_dict["corr"].append(learn_pre_list[part])
    
    data_dict["part"].append(part+1)
    data_dict["cond"].append(2)
    data_dict["run"].append(2)
    data_dict["corr"].append(learn_post_list[part])

frame = pd.DataFrame(data = data_dict)

print(AnovaRM(data = frame, depvar = "corr", subject = "part", within = ["cond", "run"]).fit()) 
