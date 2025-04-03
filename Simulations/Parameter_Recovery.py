model_code = """
data {
    real cage[50, 20, 10];
    real crab[50, 20, 10];
}
parameters {
    real alfa_mu;
    real eta_mu;
    
    real<lower=0> alfa_sigma;
    real<lower=0> eta_sigma;
    
    real alfa_prime[50];
    real eta_prime[50];
}
transformed parameters {
    real alfa[50];
    real eta[50];
    
    for (part in 1:50) {
        alfa[part] = Phi_approx(alfa_mu + alfa_sigma * alfa_prime[part]);
        eta[part] = Phi_approx(eta_mu + eta_sigma * eta_prime[part]);
    }
}
model {
    alfa_mu ~ normal(0, 1);
    eta_mu ~ normal(0, 1);
    
    alfa_sigma ~ cauchy(0, 5);
    eta_sigma ~ cauchy(0, 5);
    
    alfa_prime ~ normal(0, 1);
    eta_prime ~ normal(0, 1);
    
    for (part in 1:50) {
        for (block in 1:20) {
            
            real estim;
            real update;
            estim = 0.5;
            update = alfa[part];
            
            for (trial in 1:10) {
                cage[part, block, trial] ~ normal(estim, 0.125);
                estim += update * (crab[part, block, trial] - estim);
                update = eta[part] * fabs(crab[part, block, trial] - estim) + (1 - eta[part]) * update;
        }
        }
    }
}
generated quantities {
    real learn[50];
    real decay[50];
    
    for (part in 1:50) {
        learn[part] = Phi_approx(alfa_mu + alfa_sigma * alfa_prime[part]);
        decay[part] = Phi_approx(eta_mu + eta_sigma * eta_prime[part]);
    }
}
"""

import os
import numpy as np
import pandas as pd
import pystan

os.chdir("C:/Users/.../Experiment 2/Analyses")

def simulation(env, alfa, eta):
    
    cage_list = np.zeros(10)
    crab_list = np.zeros(10)
    
    prior_var = 0.25 - env * 0.0625
    samp_var = env * 0.0625
    samp_mean = np.random.normal(0.5, prior_var)
    cage = 0.5

    for trial in range(10):
        
        cage_list[trial] = cage
        crab = np.random.normal(samp_mean, samp_var)
        crab_list[trial] = crab
        
        error = crab - cage
        cage = cage + alfa * error
        alfa = eta * abs(error) + (1 - eta) * alfa
    
    frame = pd.DataFrame({"crab": crab_list, "cage": cage_list})
    
    return frame

for env in range(3):
    for learn in range(4):
        for decay in range(4):
            
            output = {
                "part": np.zeros(50),
                "env": np.zeros(50),
                "true_learn": np.zeros(50),
                "true_decay": np.zeros(50),
                "est_learn": np.zeros(50),
                "est_decay": np.zeros(50)
            }
        
            cage = np.zeros([50, 20, 10], dtype = float)
            crab = np.zeros([50, 20, 10], dtype = float)
            
            for part in range(50):
                
                alfa = 0
                eta = 0
                
                while alfa <= 0 or alfa >= 1:
                    alfa = np.random.normal((learn+1)*0.2, 0.2)
                while eta <= 0 or eta >= 1:
                    eta = np.random.normal((decay+1)*0.2, 0.2)
                    
                output["part"][part] = part+1
                output["env"][part] = env+1
                output["true_learn"][part] = alfa
                output["true_decay"][part] = eta
        
                for block in range(20):
                    
                    frame = simulation(env+1, alfa, eta)
                    
                    cage[part, block] = np.array(frame["cage"])
                    crab[part, block] = np.array(frame["crab"])
    
            data = {"cage": cage, "crab": crab}
            
            model = pystan.StanModel(model_code = model_code)
            trace = model.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)
            
            for part in range(50):
                output["est_learn"][part] = np.mean(trace["learn"][:, part])
                output["est_decay"][part] = np.mean(trace["decay"][:, part])
            
            output = pd.DataFrame(data = output)
            
            output.to_csv("env_{}_alfa_{}_eta_{}.csv".format(env+1, np.round((learn+1)*0.2, 1), np.round((decay+1)*0.2, 1)), index = False)
            