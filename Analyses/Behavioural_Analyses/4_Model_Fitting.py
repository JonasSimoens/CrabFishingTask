hybrid_model_code = """
data {
    int trial_count[53, 3, 40];
    real cage[53, 3, 40, 8];
    real crab[53, 3, 40, 8];
}
parameters {
    real alfa_mu[3];
    real eta_mu[3];
    
    real<lower=0> alfa_sigma[3];
    real<lower=0> eta_sigma[3];
    
    real alfa_prime[53, 3];
    real eta_prime[53, 3];
}
transformed parameters {
    real alfa[53, 3];
    real eta[53, 3];
    
    for (part in 1:53) {
        for (env in 1:3) {
            alfa[part, env] = Phi_approx(alfa_mu[env] + alfa_sigma[env] * alfa_prime[part, env]);
            eta[part, env] = Phi_approx(eta_mu[env] + eta_sigma[env] * eta_prime[part, env]);
        }
    }
}
model {
    alfa_mu ~ normal(0, 1);
    eta_mu ~ normal(0, 1);
    
    alfa_sigma ~ cauchy(0, 5);
    eta_sigma ~ cauchy(0, 5);
    
    for (part in 1:53) {
        alfa_prime[part] ~ normal(0, 1);
        eta_prime[part] ~ normal(0, 1);
    }
    
    for (part in 1:53) {
        for (env in 1:3) {
            for (block in 1:40) {
                
                real estim;
                real update;
                estim = 0.5;
                update = alfa[part, env];
                
                for (trial in 1:(trial_count[part, env, block])) {
                    cage[part, env, block, trial] ~ normal(estim, 0.125);
                    estim += update * (crab[part, env, block, trial] - estim);
                    update = eta[part, env] * fabs(crab[part, env, block, trial] - estim) + (1 - eta[part, env]) * update;
                }
            }
        }
    }
}
generated quantities {
    real learn_mu[3];
    real decay_mu[3];
    
    real learn[53, 3];
    real decay[53, 3];
    
    for (env in 1:3) {
        learn_mu[env] = Phi_approx(alfa_mu[env]);
        decay_mu[env] = Phi_approx(eta_mu[env]);
    }
    
    for (part in 1:53) {
        for (env in 1:3) {
            learn[part, env] = Phi_approx(alfa_mu[env] + alfa_sigma[env] * alfa_prime[part, env]);
            decay[part, env] = Phi_approx(eta_mu[env] + eta_sigma[env] * eta_prime[part, env]);
        }
    }
}
"""

import os
import numpy as np
import pandas as pd
import pystan
import pickle
import arviz as az

os.chdir("C:/Users/...Experiment 2/Analyses")

data = pd.read_csv("Behavioural_Data.csv")
data = data.loc[(data["trial"] != 1) & (data["trial"] != 10)].reset_index()

data["cage"] = (data["cage"] + 512) / 1024
data["crab"] = (data["crab"] + 512) / 1024

trial_count = np.zeros([53, 3, 40], dtype = int)
cage = np.zeros([53, 3, 40, 8], dtype = float)
crab = np.zeros([53, 3, 40, 8], dtype = float)

for part in range(53):
    
    count = np.zeros([3], dtype = int)
    
    for block in range(120):
        
        frame = data.loc[(data["part"] == part+1) & (data["block"] == block+1)].reset_index(drop = True)
        
        env = int(frame["sigma"][0] - 1)
        
        trial_count[part, env, count[env]] = len(frame)
        
        pre_cage = np.array(frame["cage"])
        pre_crab = np.array(frame["crab"])
        
        while len(pre_cage) < 8:
            pre_cage = np.append(pre_cage, -1)
            pre_crab = np.append(pre_crab, -1)
        
        cage[part, env, count[env]] = pre_cage
        crab[part, env, count[env]] = pre_crab
        
        count[env] = count[env] + 1

data = {"trial_count": trial_count, "cage": cage, "crab": crab}

hybrid_model = pystan.StanModel(model_code = hybrid_model_code)
hybrid_model_trace = hybrid_model.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)

check = pystan.check_hmc_diagnostics(hybrid_model_trace)

file = open("Hybrid_Model_Check.csv", "a")
for key in check.keys():
    file.write("{},{}\n".format(key, check[key]))
file.close()

file = open("Hybrid_Model_Trace.pkl", "wb")
pickle.dump({"model": hybrid_model, "trace": hybrid_model_trace}, file = file)
file.close()

output = pd.DataFrame({})
output["low_sigma_learn_mu"] = hybrid_model_trace["learn_mu"][:, 0]
output["low_sigma_decay_mu"] = hybrid_model_trace["decay_mu"][:, 0]
output["medium_sigma_learn_mu"] = hybrid_model_trace["learn_mu"][:, 1]
output["medium_sigma_decay_mu"] = hybrid_model_trace["decay_mu"][:, 1]
output["high_sigma_learn_mu"] = hybrid_model_trace["learn_mu"][:, 2]
output["high_sigma_decay_mu"] = hybrid_model_trace["decay_mu"][:, 2]

for part in range(53):
    output["low_sigma_learn_part_{}".format(part+1)] = hybrid_model_trace["learn"][:, part, 0]
    output["low_sigma_decay_part_{}".format(part+1)] = hybrid_model_trace["decay"][:, part, 0]
    output["medium_sigma_learn_part_{}".format(part+1)] = hybrid_model_trace["learn"][:, part, 1]
    output["medium_sigma_decay_part_{}".format(part+1)] = hybrid_model_trace["decay"][:, part, 1]
    output["high_sigma_learn_part_{}".format(part+1)] = hybrid_model_trace["learn"][:, part, 2]
    output["high_sigma_decay_part_{}".format(part+1)] = hybrid_model_trace["decay"][:, part, 2]

output.to_csv("Hybrid_Model_Trace.csv", index = False)
