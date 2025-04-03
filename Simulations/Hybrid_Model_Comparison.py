RW_model_code = """
data {
    real cage[50, 3, 20, 10];
    real crab[50, 3, 20, 10];
}
parameters {
    real alfa_mu[3];
    
    real<lower=0> alfa_sigma[3];
    
    real alfa_prime[50, 3];
}
transformed parameters {
    real alfa[50, 3];
    
    for (part in 1:50) {
        for (env in 1:3) {
            alfa[part, env] = Phi_approx(alfa_mu[env] + alfa_sigma[env] * alfa_prime[part, env]);
        }
    }
}
model {
    alfa_mu ~ normal(0, 1);
    
    alfa_sigma ~ cauchy(0, 5);
    
    for (part in 1:50) {
        alfa_prime[part] ~ normal(0, 1);
    }
    
    for (part in 1:50) {
        for (env in 1:3) {
            for (block in 1:20) {
                
                real estim;
                estim = 0.5;
                
                for (trial in 1:10) {
                    cage[part, env, block, trial] ~ normal(estim, 0.125);
                    estim += alfa[part, env] * (crab[part, env, block, trial] - estim);
                }
            }
        }
    }
}
generated quantities {
    real log_lik[50];
    
    {
        for (part in 1:50) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:20) {
                    
                    real estim;
                    estim = 0.5;
                    
                    for (trial in 1:10) {
                        log_lik[part] += normal_lpdf(cage[part, env, block, trial] | estim, 0.125);
                        estim += alfa[part, env] * (crab[part, env, block, trial] - estim);
                    }
                }
            }
        }
    }
}
"""

hybrid_model_code = """
data {
    real cage[50, 3, 20, 10];
    real crab[50, 3, 20, 10];
}
parameters {
    real alfa_mu[3];
    real eta_mu[3];
    
    real<lower=0> alfa_sigma[3];
    real<lower=0> eta_sigma[3];
    
    real alfa_prime[50, 3];
    real eta_prime[50, 3];
}
transformed parameters {
    real alfa[50, 3];
    real eta[50, 3];
    
    for (part in 1:50) {
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
    
    for (part in 1:50) {
        alfa_prime[part] ~ normal(0, 1);
        eta_prime[part] ~ normal(0, 1);
    }
    
    for (part in 1:50) {
        for (env in 1:3) {
            for (block in 1:20) {
                
                real estim;
                real update;
                estim = 0.5;
                update = alfa[part, env];
                
                for (trial in 1:10) {
                    cage[part, env, block, trial] ~ normal(estim, 0.125);
                    estim += update * (crab[part, env, block, trial] - estim);
                    update = eta[part, env] * fabs(crab[part, env, block, trial] - estim) + (1 - eta[part, env]) * update;
                }
            }
        }
    }
}
generated quantities {
    real log_lik[50];
    
    {
        for (part in 1:50) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:20) {
                    
                    real estim;
                    real update;
                    estim = 0.5;
                    update = alfa[part, env];
                    
                    for (trial in 1:10) {
                        log_lik[part] += normal_lpdf(cage[part, env, block, trial] | estim, 0.125);
                        estim += update * (crab[part, env, block, trial] - estim);
                        update = eta[part, env] * fabs(crab[part, env, block, trial] - estim) + (1 - eta[part, env]) * update;
                    }
                }
            }
        }
    }
}
"""

kalman_filter_code = """
data {
    real cage[50, 3, 20, 10];
    real crab[50, 3, 20, 10];
}
parameters {
    real prior_mu[3];
    
    real<lower=0> prior_sigma[3];
    
    real prior_prime[50, 3];
}
transformed parameters {
    real prior[50, 3];
    
    for (part in 1:50) {
        for (env in 1:3) {
            prior[part, env] = Phi_approx(prior_mu[env] + prior_sigma[env] * prior_prime[part, env]);
        }
    }
}
model {
    prior_mu ~ normal(0, 1);
    
    prior_sigma ~ cauchy(0, 5);
    
    for (part in 1:50) {
        prior_prime[part] ~ normal(0, 1);
    }
    
    for (part in 1:50) {
        for (env in 1:3) {
            for (block in 1:20) {
                
                real estim;
                real doubt;
                real gain;
                estim = 0.5;
                doubt = prior[part, env]^2;
                
                for (trial in 1:10) {
                    cage[part, env, block, trial] ~ normal(estim, 0.125);
                    gain = doubt / (doubt + 0.125^2);
                    estim += gain * (crab[part, env, block, trial] - estim);
                    doubt *= (1 - gain);
                }
            }
        }
    }
}
generated quantities {
    real log_lik[50];
    
    {
        for (part in 1:50) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:20) {
                    
                    real estim;
                    real doubt;
                    real gain;
                    estim = 0.5;
                    doubt = prior[part, env]^2;
                    
                    for (trial in 1:10) {
                        log_lik[part] += normal_lpdf(cage[part, env, block, trial] | estim, 0.125);
                        gain = doubt / (doubt + 0.125^2);
                        estim += gain * (crab[part, env, block, trial] - estim);
                        doubt *= (1 - gain);
                    }
                }
            }
        }
    }
}
"""

import os
import numpy as np
import pandas as pd
import pystan
import arviz as az

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

for sim in range(10):
    
    cage = np.zeros([50, 3, 20, 10], dtype = float)
    crab = np.zeros([50, 3, 20, 10], dtype = float)
    
    for env in range(3):
        
        if env == 0:
            alfa_mu = 0.8
        if env == 1:
            alfa_mu = 0.65
        if env == 2:
            alfa_mu = 0.45
        
        for part in range(50):
            
            alfa = 0
            eta = 0
            
            while alfa <= 0 or alfa >= 1:
                alfa = np.random.normal(alfa_mu, 0.2)
            while eta <= 0 or eta >= 1:
                eta = np.random.normal(0, 0.2)
    
            for block in range(20):
                
                frame = simulation(env+1, alfa, eta)
                
                cage[part, env, block] = np.array(frame["cage"])
                crab[part, env, block] = np.array(frame["crab"])

    data = {"cage": cage, "crab": crab}
    
    RW_model = pystan.StanModel(model_code = RW_model_code)
    RW_model_trace = RW_model.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)
    
    hybrid_model = pystan.StanModel(model_code = hybrid_model_code)
    hybrid_model_trace = hybrid_model.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)
    
    kalman_filter = pystan.StanModel(model_code = kalman_filter_code)
    kalman_filter_trace = kalman_filter.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)
    
    dataset_dict = {"RW_model": az.from_pystan(RW_model_trace, log_likelihood = "log_lik"),
                    "hybrid_model": az.from_pystan(hybrid_model_trace, log_likelihood = "log_lik"),
                    "kalman_filter": az.from_pystan(kalman_filter_trace, log_likelihood = "log_lik")}

    looic_comp = az.compare(dataset_dict = dataset_dict, ic = "loo")

    looic_comp = pd.DataFrame(looic_comp)
    looic_comp.to_csv("Hybrid_Model_Comparison_{}.csv".format(sim+1))
    