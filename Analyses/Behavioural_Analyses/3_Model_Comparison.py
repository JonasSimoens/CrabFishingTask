RW_model_code = """
data {
    int trial_count[53, 3, 40];
    real cage[53, 3, 40, 8];
    real crab[53, 3, 40, 8];
}
parameters {
    real alfa_mu[3];
    
    real<lower=0> alfa_sigma[3];
    
    real alfa_prime[53, 3];
}
transformed parameters {
    real alfa[53, 3];
    
    for (part in 1:53) {
        for (env in 1:3) {
            alfa[part, env] = Phi_approx(alfa_mu[env] + alfa_sigma[env] * alfa_prime[part, env]);
        }
    }
}
model {
    alfa_mu ~ normal(0, 1);
    
    alfa_sigma ~ cauchy(0, 5);
    
    for (part in 1:53) {
        alfa_prime[part] ~ normal(0, 1);
    }
    
    for (part in 1:53) {
        for (env in 1:3) {
            for (block in 1:40) {
                
                real estim;
                estim = 0.5;
                
                for (trial in 1:(trial_count[part, env, block])) {
                    cage[part, env, block, trial] ~ normal(estim, 0.125);
                    estim += alfa[part, env] * (crab[part, env, block, trial] - estim);
                }
            }
        }
    }
}
generated quantities {
    real log_lik[53];
    
    {
        for (part in 1:53) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:40) {
                    
                    real estim;
                    estim = 0.5;
                    
                    for (trial in 1:(trial_count[part, env, block])) {
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
    real log_lik[53];
    
    {
        for (part in 1:53) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:40) {
                    
                    real estim;
                    real update;
                    estim = 0.5;
                    update = alfa[part, env];
                    
                    for (trial in 1:(trial_count[part, env, block])) {
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
    int trial_count[53, 3, 40];
    real cage[53, 3, 40, 8];
    real crab[53, 3, 40, 8];
}
parameters {
    real prior_mu[3];
    
    real<lower=0> prior_sigma[3];
    
    real prior_prime[53, 3];
}
transformed parameters {
    real prior[53, 3];
    
    for (part in 1:53) {
        for (env in 1:3) {
            prior[part, env] = Phi_approx(prior_mu[env] + prior_sigma[env] * prior_prime[part, env]);
        }
    }
}
model {
    prior_mu ~ normal(0, 1);
    
    prior_sigma ~ cauchy(0, 5);
    
    for (part in 1:53) {
        prior_prime[part] ~ normal(0, 1);
    }
    
    for (part in 1:53) {
        for (env in 1:3) {
            for (block in 1:40) {
                
                real estim;
                real doubt;
                real gain;
                estim = 0.5;
                doubt = prior[part, env]^2;
                
                for (trial in 1:(trial_count[part, env, block])) {
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
    real log_lik[53];
    
    {
        for (part in 1:53) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:40) {
                    
                    real estim;
                    real doubt;
                    real gain;
                    estim = 0.5;
                    doubt = prior[part, env]^2;
                    
                    for (trial in 1:(trial_count[part, env, block])) {
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

rest_RW_model_code = """
data {
    int trial_count[53, 3, 40];
    real cage[53, 3, 40, 8];
    real crab[53, 3, 40, 8];
}
parameters {
    real alfa_mu;
    
    real<lower=0> alfa_sigma;
    
    real alfa_prime[53];
}
transformed parameters {
    real alfa[53];
    
    for (part in 1:53) {
        alfa[part] = Phi_approx(alfa_mu + alfa_sigma * alfa_prime[part]);
    }
}
model {
    alfa_mu ~ normal(0, 1);
    
    alfa_sigma ~ cauchy(0, 5);
    
    alfa_prime ~ normal(0, 1);
    
    for (part in 1:53) {
        for (env in 1:3) {
            for (block in 1:40) {
                
                real estim;
                estim = 0.5;
                
                for (trial in 1:(trial_count[part, env, block])) {
                    cage[part, env, block, trial] ~ normal(estim, 0.125);
                    estim += alfa[part] * (crab[part, env, block, trial] - estim);
                }
            }
        }
    }
}
generated quantities {
    real log_lik[53];
    
    {
        for (part in 1:53) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:40) {
                    
                    real estim;
                    estim = 0.5;
                    
                    for (trial in 1:(trial_count[part, env, block])) {
                        log_lik[part] += normal_lpdf(cage[part, env, block, trial] | estim, 0.125);
                        estim += alfa[part] * (crab[part, env, block, trial] - estim);
                    }
                }
            }
        }
    }
}
"""

rest_hybrid_model_code = """
data {
    int trial_count[53, 3, 40];
    real cage[53, 3, 40, 8];
    real crab[53, 3, 40, 8];
}
parameters {
    real alfa_mu;
    real eta_mu;
    
    real<lower=0> alfa_sigma;
    real<lower=0> eta_sigma;
    
    real alfa_prime[53];
    real eta_prime[53];
}
transformed parameters {
    real alfa[53];
    real eta[53];
    
    for (part in 1:53) {
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
    
    for (part in 1:53) {
        for (env in 1:3) {
            for (block in 1:40) {
                
                real estim;
                real update;
                estim = 0.5;
                update = alfa[part];
                
                for (trial in 1:(trial_count[part, env, block])) {
                    cage[part, env, block, trial] ~ normal(estim, 0.125);
                    estim += update * (crab[part, env, block, trial] - estim);
                    update = eta[part] * fabs(crab[part, env, block, trial] - estim) + (1 - eta[part]) * update;
                }
            }
        }
    }
}
generated quantities {
    real log_lik[53];
    
    {
        for (part in 1:53) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:40) {
                    
                    real estim;
                    real update;
                    estim = 0.5;
                    update = alfa[part];
                    
                    for (trial in 1:(trial_count[part, env, block])) {
                        log_lik[part] += normal_lpdf(cage[part, env, block, trial] | estim, 0.125);
                        estim += update * (crab[part, env, block, trial] - estim);
                        update = eta[part] * fabs(crab[part, env, block, trial] - estim) + (1 - eta[part]) * update;
                    }
                }
            }
        }
    }
}
"""

rest_kalman_filter_code = """
data {
    int trial_count[53, 3, 40];
    real cage[53, 3, 40, 8];
    real crab[53, 3, 40, 8];
}
parameters {
    real prior_mu;
    
    real<lower=0> prior_sigma;
    
    real prior_prime[53];
}
transformed parameters {
    real prior[53];
    
    for (part in 1:53) {
        prior[part] = Phi_approx(prior_mu + prior_sigma * prior_prime[part]);
    }
}
model {
    prior_mu ~ normal(0, 1);
    
    prior_sigma ~ cauchy(0, 5);
    
    prior_prime ~ normal(0, 1);
    
    for (part in 1:53) {
        for (env in 1:3) {
            for (block in 1:40) {
                
                real estim;
                real doubt;
                real gain;
                estim = 0.5;
                doubt = prior[part]^2;
                
                for (trial in 1:(trial_count[part, env, block])) {
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
    real log_lik[53];
    
    {
        for (part in 1:53) {
            log_lik[part] = 0;
            for (env in 1:3) {
                for (block in 1:40) {
                    
                    real estim;
                    real doubt;
                    real gain;
                    estim = 0.5;
                    doubt = prior[part]^2;
                    
                    for (trial in 1:(trial_count[part, env, block])) {
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

RW_model = pystan.StanModel(model_code = RW_model_code)
RW_model_trace = RW_model.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)

check = pystan.check_hmc_diagnostics(RW_model_trace)

file = open("RW_Model_Check.csv", "a")
for key in check.keys():
    file.write("{},{}\n".format(key, check[key]))
file.close()

file = open("RW_Model_Trace.pkl", "wb")
pickle.dump({"model": RW_model, "trace": RW_model_trace}, file = file)
file.close()

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

kalman_filter = pystan.StanModel(model_code = kalman_filter_code)
kalman_filter_trace = kalman_filter.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)

check = pystan.check_hmc_diagnostics(kalman_filter_trace)

file = open("Kalman_Filter_Check.csv", "a")
for key in check.keys():
    file.write("{},{}\n".format(key, check[key]))
file.close()

file = open("Kalman_Filter_Trace.pkl", "wb")
pickle.dump({"model": kalman_filter, "trace": kalman_filter_trace}, file = file)
file.close()

rest_RW_model = pystan.StanModel(model_code = rest_RW_model_code)
rest_RW_model_trace = rest_RW_model.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)

check = pystan.check_hmc_diagnostics(rest_RW_model_trace)

file = open("Rest_RW_Model_Check.csv", "a")
for key in check.keys():
    file.write("{},{}\n".format(key, check[key]))
file.close()

file = open("Rest_RW_Model_Trace.pkl", "wb")
pickle.dump({"model": rest_RW_model, "trace": rest_RW_model_trace}, file = file)
file.close()

rest_hybrid_model = pystan.StanModel(model_code = rest_hybrid_model_code)
rest_hybrid_model_trace = rest_hybrid_model.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)

check = pystan.check_hmc_diagnostics(rest_hybrid_model_trace)

file = open("Rest_Hybrid_Model_Check.csv", "a")
for key in check.keys():
    file.write("{},{}\n".format(key, check[key]))
file.close()

file = open("Rest_Hybrid_Model_Trace.pkl", "wb")
pickle.dump({"model": rest_hybrid_model, "trace": rest_hybrid_model_trace}, file = file)
file.close()

rest_kalman_filter = pystan.StanModel(model_code = rest_kalman_filter_code)
rest_kalman_filter_trace = rest_kalman_filter.sampling(data = data, chains = 4, iter = 4000, warmup = 1000)

check = pystan.check_hmc_diagnostics(rest_kalman_filter_trace)

file = open("Rest_Kalman_Filter_Check.csv", "a")
for key in check.keys():
    file.write("{},{}\n".format(key, check[key]))
file.close()

file = open("Rest_Kalman_Filter_Trace.pkl", "wb")
pickle.dump({"model": rest_kalman_filter, "trace": rest_kalman_filter_trace}, file = file)
file.close()

dataset_dict = {"RW_model": az.from_pystan(RW_model_trace, log_likelihood = "log_lik"),
    "hybrid_model": az.from_pystan(hybrid_model_trace, log_likelihood = "log_lik"),
    "kalman_filter": az.from_pystan(kalman_filter_trace, log_likelihood = "log_lik"),
    "rest_RW_model": az.from_pystan(rest_RW_model_trace, log_likelihood = "log_lik"),
    "rest_hybrid_model": az.from_pystan(rest_hybrid_model_trace, log_likelihood = "log_lik"),
    "rest_kalman_filter": az.from_pystan(rest_kalman_filter_trace, log_likelihood = "log_lik")}

looic_comp = az.compare(dataset_dict = dataset_dict, ic = "loo")

looic_comp = pd.DataFrame(looic_comp)
looic_comp.to_csv("Model_Comparison.csv")
