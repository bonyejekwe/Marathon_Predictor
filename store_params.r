# remove.packages(c("StanHeaders", "rstan"))
# if (file.exists(".RData")) file.remove(".RData")
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# example(stan_model, package = "rstan", run.dontrun = TRUE)
# install.packages("loo")

# getwd()
set.seed(2025)
library("rstan") # observe startup messages
library("loo")
features1 <- c("total_pace")
features2 <- c("total_pace", "curr_ratio")
features3 <- c("total_pace") # w/ curr_pace
features4 <- c("total_pace", "male", "age_t") # w/ curr_pace

for (race in c("all")) {
    print(race)
    train_name <- paste("processed_data/train_", race, ".csv", sep="")
    train_data <- read.csv(train_name)
    
    data1 <- list(N = nrow(train_data), K = length(features1), L = 8, R=3,
                  feats = train_data[features1], split_idx = train_data$lvl,
                  marathon_idx = train_data$race_val, finish = train_data$finish)
    res_name1 <- paste("stan_results/result_", race, "1.csv", sep="")
    par_name1 <- paste("stan_results/params_", race, "1.csv", sep="")
    fit1 <- stan(file = 'marathon.stan', data = data1,
                 iter=800, chains=4, cores=4, seed=2025,
                 control = list(max_treedepth = 12))
    parameters1 <- as.data.frame(extract(fit1)[c("beta", "sigma", "lp__")])
    write.csv(parameters1, par_name1, row.names = TRUE)
    check_hmc_diagnostics(fit1)
    llk1 <- extract_log_lik(fit1, parameter_name = "log_lik", merge_chains = TRUE)
    loo1 <- loo(llk1)
    print(loo1)

    data2 <- list(N = nrow(train_data), K = length(features2), L = 8, R=3,
                  feats = train_data[features2], split_idx = train_data$lvl,
                  marathon_idx = train_data$race_val, finish = train_data$finish)
    res_name2 <- paste("stan_results/result_", race, "2.csv", sep="")
    par_name2 <- paste("stan_results/params_", race, "2.csv", sep="")
    fit2 <- stan(file = 'marathon.stan', data = data2,
                 iter=800, chains=4, cores=4, seed=2025,
                 control = list(max_treedepth = 12))
    parameters2 <- as.data.frame(extract(fit2)[c("beta", "sigma", "lp__")])
    write.csv(parameters2, par_name2, row.names = TRUE)
    check_hmc_diagnostics(fit2)
    llk2 <- extract_log_lik(fit2, parameter_name = "log_lik", merge_chains = TRUE)
    loo2 <- loo(llk2)
    print(loo2)
    wsws
    data3 <- list(N = nrow(train_data), K = length(features3), L = 8, R=3, marathon_idx = train_data$race_val,
                  feats = train_data[features3], split_idx = train_data$lvl,
                  curr = train_data$curr_pace,
                  finish = train_data$finish)
    res_name3 <- paste("stan_results/result_", race, "3.csv", sep="")
    par_name3 <- paste("stan_results/params_", race, "3.csv", sep="")
    fit3 <- stan(file = 'marathon2.stan', data = data3,
                 iter=800, chains=4, cores=4, seed=2025,
                 control = list(max_treedepth = 12))
    parameters3 <- as.data.frame(extract(fit3)[c("beta", "gamma", "sigma", "lp__")])
    write.csv(parameters3, par_name3, row.names = TRUE)
    check_hmc_diagnostics(fit3)
    llk3 <- extract_log_lik(fit3, parameter_name = "log_lik", merge_chains = TRUE)
    loo3 <- loo(llk3)
    print(loo3)

    data4 <- list(N = nrow(train_data), K = length(features4), L = 8, R=3, marathon_idx = train_data$race_val,
                  feats = train_data[features4], split_idx = train_data$lvl,
                  curr = train_data$curr_pace,
                  finish = train_data$finish)
    res_name4 <- paste("stan_results/result_", race, "4.csv", sep="")
    par_name4 <- paste("stan_results/params_", race, "4.csv", sep="")
    fit4 <- stan(file = 'marathon2.stan', data = data4,
                 iter=800, chains=4, cores=4, seed=2025,
                 control = list(max_treedepth = 12))
    parameters4 <- as.data.frame(extract(fit4)[c("beta", "gamma", "sigma", "lp__")])
    write.csv(parameters4, par_name4, row.names = TRUE)
    check_hmc_diagnostics(fit4)
    llk4 <- extract_log_lik(fit4, parameter_name = "log_lik", merge_chains = TRUE)
    loo4 <- loo(llk4)
    print(loo4)
    
    comp <- loo_compare(loo1, loo2, loo3, loo4)
    print(comp, simplify=FALSE)
    # stan_rhat(fit3)
    # traceplot(fit3, inc_warmup=TRUE)
    # stan_ess(fit3)
}

# saveRDS(fit, file = "my_stan_fit.rds")
# print(names(extract(fit)))
# x = extract(fit)["finish_test"]
# summary(fit, pars=c('alpha', "beta", 'sigma'), probs=c(.25, .75))$summary
