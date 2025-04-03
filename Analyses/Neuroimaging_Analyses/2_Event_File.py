import os
import pandas as pd
import numpy as np

os.chdir("C:/Users/.../Experiment 2/Analyses")

input_data = pd.read_csv("Behavioural_Data.csv")

data = {
    "part": [],
    "run": [],
    "angle": [],
    "event": [],
    "onset": [],
    "error": []
}

for part in range(53):
    
    frame = input_data.loc[input_data["part"] == part+1].reset_index()
    
    for index in range(780):
        
        if frame["trial"][index] == 1:
            
            data["part"].append(part+1)
            data["run"].append(frame["run"][index])
            data["angle"].append(frame["angle"][index])
            data["event"].append(1)
            data["onset"].append(frame["blockTime"][index] - 1.78/2)
            data["error"].append(-1)
            
        if frame["trial"][index] == 2:
            
            data["part"].append(part+1)
            data["run"].append(frame["run"][index])
            data["angle"].append(frame["angle"][index])
            data["event"].append(2)
            data["onset"].append(frame["blockTime"][index] - 1.78/2)
            data["error"].append(-1)
            
            data["part"].append(part+1)
            data["run"].append(frame["run"][index])
            data["angle"].append(frame["angle"][index])
            data["event"].append(3)
            data["onset"].append(frame["blockTime"][index] + frame["respTime"][index] + 0.75 - 1.78/2)
            data["error"].append(frame["error"][index])
            
        if frame["trial"][index] == 3:
            
            data["part"].append(part+1)
            data["run"].append(frame["run"][index])
            data["angle"].append(frame["angle"][index])
            data["event"].append(4)
            data["onset"].append(frame["blockTime"][index] - 1.78/2)
            data["error"].append(-1)
            
            data["part"].append(part+1)
            data["run"].append(frame["run"][index])
            data["angle"].append(frame["angle"][index])
            data["event"].append(5)
            data["onset"].append(frame["blockTime"][index] + frame["respTime"][index] + 0.75 - 1.78/2)
            data["error"].append(frame["error"][index])
        
        if (frame["trial"][index] > 3) & (frame["trial"][index] < 10):
            
            data["part"].append(part+1)
            data["run"].append(frame["run"][index])
            data["angle"].append(frame["angle"][index])
            data["event"].append(6)
            data["onset"].append(frame["blockTime"][index] - 1.78/2)
            data["error"].append(-1)
            
        if frame["trial"][index] == 10:
            
            data["part"].append(part+1)
            data["run"].append(frame["run"][index])
            data["angle"].append(frame["angle"][index])
            data["event"].append(7)
            data["onset"].append(frame["blockTime"][index] - 1.78/2)
            data["error"].append(-1)
            
data = pd.DataFrame(data = data)

for part in range(53):
    for run in range(4):
        for angle in range(6):
                
            mean = np.mean(data.loc[(data["part"] == part+1) & (data["run"] == run+1) & (data["angle"] == angle+1) & (data["event"] == 3)]["error"])
            data["error"][(data["part"] == part+1) & (data["run"] == run+1) & (data["angle"] == angle+1) & (data["event"] == 3)] = data["error"] - mean
            
            mean = np.mean(data.loc[(data["part"] == part+1) & (data["run"] == run+1) & (data["angle"] == angle+1) & (data["event"] == 5)]["error"])
            data["error"][(data["part"] == part+1) & (data["run"] == run+1) & (data["angle"] == angle+1) & (data["event"] == 5)] = data["error"] - mean

data.to_csv("Event_File.csv", index = False)
