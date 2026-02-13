# remove.packages(c("StanHeaders", "rstan"))
# if (file.exists(".RData")) file.remove(".RData")
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# example(stan_model, package = "rstan", run.dontrun = TRUE)

# getwd()
set.seed(2025)
library("rstan") # observe startup messages

features1 <- c("alpha", "total_pace")
features2 <- c("alpha", "total_pace", "curr_pace")
features3 <- c("alpha", "total_pace", "curr_pace", "male", "age")
feat_list = list(features1, features2, features3)

for (num in 1:3) {
  for (race in c("bos", "nyc", "chi")) {
    print(race)
    print(num)
    features <- feat_list[[num]]
    print(features)
    train_name <- paste("processed_data/train_", race, ".csv", sep="")
    res_name <- paste("stan_results/model", num, "/result_", race, ".csv", sep="")
    par_name <- paste("stan_results/model", num, "/params_", race, ".csv", sep="")
  
    train_data <- read.csv(train_name)
    s_dat <- list(N = nrow(train_data),
                  K = length(features),
                  L = 8,
                  feats = train_data[features],
                  ll = train_data$lvl,
                  finish = train_data$finish)
  
    # time the stan fit and record elapsed seconds
    rt_file <- paste0("stan_results/model", num, "/runtimes.csv")
    elapsed <- as.numeric(system.time(fit <- stan(file = 'marathon.stan', data = s_dat, cores=4, seed=2025))["elapsed"])
    rt_row <- data.frame(race = race, model = num, elapsed = elapsed)
    if (!file.exists(rt_file)) {
      write.table(rt_row, file = rt_file, sep = ",", row.names = FALSE, col.names = TRUE)
    } else {
      write.table(rt_row, file = rt_file, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
    }
    parameters <- as.data.frame(extract(fit)[c("beta", "sigma", "lp__")])
    write.csv(parameters,par_name, row.names = TRUE)
  }
}

# fit
# saveRDS(fit, file = "my_stan_fit.rds")
# print(names(extract(fit)))
# x = extract(fit)["finish_test"]
# summary(fit, pars=c('alpha', "beta", 'sigma'), probs=c(.25, .75))$summary
