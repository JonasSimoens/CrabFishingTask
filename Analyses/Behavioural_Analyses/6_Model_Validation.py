import os
import pandas as pd
import numpy as np
import scipy.stats as stats

def simulation(data, start_alfa, eta):
    
    block_list = data.block.unique()
    
    count = 0
    true_resp_list = np.zeros(200)
    sim_resp_list = np.zeros(200)
    
    for block in block_list:
        
        frame = data.loc[data["block"] == block].reset_index()
        
        estimate = 0.5
        alfa = start_alfa

        for trial in range(len(frame)):
            
            true_resp_list[count] = frame["cage"][trial]
            sim_resp_list[count] = estimate
            feedback = frame["crab"][trial]
            error = feedback - estimate
            estimate = estimate + alfa * error
            alfa = eta * abs(error) + (1 - eta) * alfa
            count = count + 1
            
    corr = stats.pearsonr(true_resp_list, sim_resp_list)[0]
    
    return corr

os.chdir("C:/Users/...Experiment 2/Analyses")

data = pd.read_csv("Behavioural_Data.csv")
data = data.loc[(data["trial"] != 1) & (data["trial"] != 10)].reset_index()

data["cage"] = (data["cage"] + 512) / 1024
data["crab"] = (data["crab"] + 512) / 1024

trace = pd.read_csv("Hybrid_Model_Trace.csv")

corr_list = np.zeros([53, 3], dtype = float)

for part in range(53):
    for env, word in zip(range(3), ["low", "medium", "high"]):
        
        frame = data.loc[(data["part"] == part+1) & (data["sigma"] == (env+1))]
            
        alfa = np.mean(trace["{}_sigma_learn_part_{}".format(word, part+1)])
        eta = np.mean(trace["{}_sigma_decay_part_{}".format(word,part+1)])
        
        corr_list[part, env] = simulation(frame, alfa, eta)

print(np.mean(corr_list[:, 0]))
print(np.mean(corr_list[:, 1]))
print(np.mean(corr_list[:, 2]))
