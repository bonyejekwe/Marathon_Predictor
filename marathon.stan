data {
  int<lower=1> N;
  int<lower=1> R;
  int<lower=1> K;
  int<lower=1> L;
  matrix[N, K] feats;
  vector[N] finish;
  array[N] int<lower=1, upper=L> split_idx;
  array[N] int<lower=1, upper=R> marathon_idx;
}
parameters {
  array[L] vector[R] alpha;
  array[L] vector[K] beta;
  vector<lower=0>[L] sigma; 
}
model {
  vector[N] mu;
  // priors
  for (l in 1:L) {
    alpha[l] ~ normal(0, 1);
    beta[l] ~ normal(0, 1);
    sigma[l] ~ cauchy(0, 1);
  }
  for (n in 1:N) {
    mu[n] = alpha[split_idx[n], marathon_idx[n]] + feats[n] * beta[split_idx[n]];
  }
  finish ~ normal(mu, sigma[split_idx]);
}

generated quantities {
  vector[N] log_lik;
  vector[N] mu_hat;
  
  for (n in 1:N) {
    mu_hat[n] = alpha[split_idx[n], marathon_idx[n]] + feats[n] * beta[split_idx[n]];
    log_lik[n] = normal_lpdf(finish[n] | mu_hat[n], sigma[split_idx[n]]);
  }
}
