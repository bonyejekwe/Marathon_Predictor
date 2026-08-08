data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=1> L;
  matrix[N, K] feats;
  vector[N] finish;
  array[N] int<lower=1, upper=L> split_idx;
}
parameters {
  array[L] vector[K] beta;
  array[L] real<lower=0> sigma;
}
model {
  vector[N] mu;
  // priors
  for (l in 1:L) {
    beta[l] ~ normal(0, 1);
    sigma[l] ~ normal(0, 1); 
  }
  // model
  int block_size = N %/% L;
  for (s in 1:L) {
    int a = (s - 1) * block_size + 1;
    int b = s * block_size;
    mu[a:b] = feats[a:b] * beta[s];
  }
  finish ~ normal(mu, sigma[split_idx]);
}

generated quantities {
  vector[N] log_lik;
  vector[N] mu_hat;
  int block_size = N %/% L;
  for (s in 1:L) {
    int a = (s - 1) * block_size + 1;
    int b = s * block_size;
    mu_hat[a:b] = feats[a:b] * beta[s];
  }
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(finish[n] | mu_hat[n], sigma[split_idx[n]]);
  }
}