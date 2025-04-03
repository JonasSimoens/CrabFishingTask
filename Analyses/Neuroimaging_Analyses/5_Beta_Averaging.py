import os
import numpy as np
import pandas as pd

os.chdir("C:/Users/.../Experiment 2/Analyses/beta_values")

part_list = np.concatenate((range(1, 12), range(13, 20), range(21, 26), range(27, 52), range(53, 54)))

for part in part_list:
    
    frame = pd.read_csv("beta_values_part_{}.csv".format(part))
    
    for run in range(4):
    
        frame["mean_run_{}".format(run+1)] = (frame["angle_1_run_{}".format(run+1)] + frame["angle_2_run_{}".format(run+1)] + frame["angle_3_run_{}".format(run+1)] + frame["angle_4_run_{}".format(run+1)] + frame["angle_5_run_{}".format(run+1)] + frame["angle_6_run_{}".format(run+1)]) / 6
    
        for angle in range(6):
        
            frame["angle_{}_run_{}".format(angle+1, run+1)] = frame["angle_{}_run_{}".format(angle+1, run+1)] - frame["mean_run_{}".format(run+1)]
    
    frame.to_csv("beta_values_part_{}.csv".format(part), index = False)
