data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=1> L;
  matrix[N, K] feats;
  vector[N] finish;
  vector[N] curr;
  array[N] int<lower=1, upper=L> split_idx;
}
parameters {
  array[L] vector[K] beta;
  array[L-1] real<lower=0> gamma;
  array[L] real<lower=0> sigma;
}
model {
  vector[N] mu;
  // priors
  for (l in 1:L) {
    beta[l] ~ normal(0, 1);
    sigma[l] ~ normal(0, 1); 
  }
  for (l in 1:L-1) {
    gamma[l] ~ normal(0, 1);
  }
  // model
  int block_size = N %/% L;

  for (s in 1:L) {
    int a = (s - 1) * block_size + 1;
    int b = s * block_size;

    if (s == 1) {
      mu[a:b] = feats[a:b] * beta[s];
    } else {
      mu[a:b] = feats[a:b] * beta[s] + curr[a:b] * gamma[s-1];
    }
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

    if (s == 1) {
      mu_hat[a:b] = feats[a:b] * beta[s];
    } else {
      mu_hat[a:b] = feats[a:b] * beta[s] + curr[a:b] * gamma[s-1];
    }
  }
  for (n in 1:N) {
    log_lik[n] = normal_lpdf(finish[n] | mu_hat[n], sigma[split_idx[n]]);
  }
}