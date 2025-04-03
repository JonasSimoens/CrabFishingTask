import os
import pandas as pd

data = pd.DataFrame()

os.chdir("C:/Users/...Experiment 2/Data")

for part in range(53):
    
    frame = pd.read_csv("Behavioural_Data_Participant_{}.csv".format(part+1))
    
    for index in range(len(frame)):
        if frame["trial"][index] == 0:
            frame["sigma"][index-1] = frame["sigma"][index]
        if frame["trial"][index] == 7:
            frame["sigma"][index+1] = frame["sigma"][index]
            frame["trial"][index+1] = 8
            
    frame["part"] = frame["part"] - 1
    
    frame["run"] = -1
    frame["run"][frame["block"] < 30] = 1
    frame["run"][(frame["block"] >= 30) & (frame["block"] < 60)] = 2
    frame["run"][(frame["block"] >= 60) & (frame["block"] < 90)] = 3
    frame["run"][frame["block"] >= 90] = 4
    
    frame["block"] = frame["block"] + 1
    
    frame["angle"] = frame["angle"] / 60 + 1
    
    frame["sigma"] = frame["sigma"] / 64
    
    frame["trial"] = frame["trial"] + 2
    
    frame["error"] = -1.0
    
    for index in range(len(frame)):
        if frame["trial"][index] >= 2 & frame["trial"][index] <= 9:
            frame["error"][index] = abs(frame["cage"][index] - frame["crab"][index])
    
    data = pd.concat([data, frame])
    
data = data[["part", "run", "block", "angle", "sigma", "trial", "cage", "crab", "error", "reward", "respTime", "scanTime", "blockTime"]]
 
data = data.astype({"angle": int, "sigma": int})
    
os.chdir("C:/Users/...Experiment 2/Analyses")

data.to_csv("Behavioural_Data.csv", index = False)
