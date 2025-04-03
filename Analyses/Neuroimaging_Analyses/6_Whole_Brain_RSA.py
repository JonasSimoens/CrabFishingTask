import os
import numpy as np
import pandas as pd
import scipy.stats as st

os.chdir("C:/Users/.../Experiment 2/Analyses")

part_list = np.concatenate((range(1, 12), range(13, 20), range(21, 26), range(27, 52), range(53, 54)))

geometric_RDM = np.array([[0, 1, 2, 3, 2, 1],
                          [1, 0, 1, 2, 3, 2],
                          [2, 1, 0, 1, 2, 3],
                          [3, 2, 1, 0, 1, 2],
                          [2, 3, 2, 1, 0, 1],
                          [1, 2, 3, 2, 1, 0]])

geometric_RDV = np.array([])

row = 0
run = 4

for loop in range(18):
    
    if loop % 6 == 0:
        
        row = 0
        run = run-1
        
    geometric_RDV = np.concatenate([geometric_RDV, np.tile(geometric_RDM[row, :], run)])
    
    row = row+1

for part in part_list:
    
    input_data = pd.read_csv("beta_values/beta_values_part_{}.csv".format(part))
    
    neural_data = np.zeros([78, 93, 78, 24])
    neural_data[:] = np.nan
    
    for vox in range(len(input_data)):
        neural_data[input_data.iloc[vox, 0], input_data.iloc[vox, 1], input_data.iloc[vox, 2], :] = np.array([input_data.iloc[vox, 3:27]])
    
    x_values = []
    y_values = []
    z_values = []
    
    r_values = []
    
    for vox_x in range(78):
        for vox_y in range(93):
            for vox_z in range(78):
                
                if np.isnan(neural_data[vox_x, vox_y, vox_z, 0]):
                    continue
                    
                size_check = neural_data[vox_x-3:vox_x+4, vox_y-3:vox_y+4, vox_z-3:vox_z+4, 0].reshape((-1,))
                size_check = size_check[~np.isnan(size_check)]
                    
                if len(size_check) < 171:
                    continue
                
                neural_RDV = []
                
                run = 0
                
                for row in range(18):
                    
                    if row % 6 == 0:
                        
                        run = run+1
                    
                    data_row = neural_data[vox_x-3:vox_x+4, vox_y-3:vox_y+4, vox_z-3:vox_z+4, row].reshape((-1,))
                    data_row = data_row[~np.isnan(data_row)]
                    
                    for col in range(run*6, 24):
                    
                        data_col = neural_data[vox_x-3:vox_x+4, vox_y-3:vox_y+4, vox_z-3:vox_z+4, col].reshape((-1,))
                        data_col = data_col[~np.isnan(data_col)]
                        
                        neural_RDV.append(1 - st.pearsonr(data_row, data_col)[0])
                    
                x_values.append(vox_x)
                y_values.append(vox_y)
                z_values.append(vox_z)
                
                r_values.append(st.kendalltau(geometric_RDV, neural_RDV)[0])
    
    output = {'x': x_values, 'y': y_values, 'z': z_values, 'r': r_values}
    
    output = pd.DataFrame.from_dict(output)
    
    output.to_csv("tau_values_loc/tau_values_part_{}.csv".format(part))
    
#%%

behavioural_data = pd.read_csv("Behavioural_Data.csv")

part_list = np.concatenate((range(1, 12), range(13, 20), range(21, 26), range(27, 52), range(53, 54)))

for part in part_list:
    
    behavioural_RDM = []
    
    for row in range(6):
        
        data_row = behavioural_data.loc[(behavioural_data["part"] == part) & (behavioural_data["angle"] == row+1)].reset_index()
        data_row = data_row["sigma"][0]
        
        for col in range(6):
        
            data_col = behavioural_data.loc[(behavioural_data["part"] == part) & (behavioural_data["angle"] == col+1)].reset_index()
            data_col = data_col["sigma"][0]
            
            behavioural_RDM.append(abs(data_row - data_col))
                
    behavioural_RDM = np.asarray(behavioural_RDM)
    behavioural_RDM = behavioural_RDM.reshape(6, 6)

    behavioural_RDV = np.array([])
    
    row = 0
    run = 4
    
    for loop in range(18):
        
        if loop % 6 == 0:
            
            row = 0
            run = run-1
            
        behavioural_RDV = np.concatenate([behavioural_RDV, np.tile(behavioural_RDM[row, :], run)])
        
        row = row+1
    
    input_data = pd.read_csv("beta_values/beta_values_part_{}.csv".format(part))
    
    neural_data = np.zeros([78, 93, 78, 24])
    neural_data[:] = np.nan
    
    for vox in range(len(input_data)):
        neural_data[input_data.iloc[vox, 0], input_data.iloc[vox, 1], input_data.iloc[vox, 2], :] = np.array([input_data.iloc[vox, 3:27]])
    
    x_values = []
    y_values = []
    z_values = []
    
    r_values = []
    
    for vox_x in range(78):
        for vox_y in range(93):
            for vox_z in range(78):
                
                if np.isnan(neural_data[vox_x, vox_y, vox_z, 0]):
                    continue
                    
                size_check = neural_data[vox_x-3:vox_x+4, vox_y-3:vox_y+4, vox_z-3:vox_z+4, 0].reshape((-1,))
                size_check = size_check[~np.isnan(size_check)]
                    
                if len(size_check) < 171:
                    continue
                
                neural_RDV = []
                
                run = 0
                
                for row in range(18):
                    
                    if row % 6 == 0:
                        
                        run = run+1
                    
                    data_row = neural_data[vox_x-3:vox_x+4, vox_y-3:vox_y+4, vox_z-3:vox_z+4, row].reshape((-1,))
                    data_row = data_row[~np.isnan(data_row)]
                    
                    for col in range(run*6, 24):
                        
                        data_col = neural_data[vox_x-3:vox_x+4, vox_y-3:vox_y+4, vox_z-3:vox_z+4, col].reshape((-1,))
                        data_col = data_col[~np.isnan(data_col)]
                        
                        neural_RDV.append(1 - st.pearsonr(data_row, data_col)[0])
                    
                x_values.append(vox_x)
                y_values.append(vox_y)
                z_values.append(vox_z)
                
                r_values.append(st.kendalltau(behavioural_RDV, neural_RDV)[0])
    
    output = {'x': x_values, 'y': y_values, 'z': z_values, 'r': r_values}
    
    output = pd.DataFrame.from_dict(output)
    
    output.to_csv("tau_values_learn/tau_values_part_{}.csv".format(part))
