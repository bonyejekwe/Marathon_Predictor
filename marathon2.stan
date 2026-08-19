data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=1> L;
  matrix[N, K] feats;
  vector[N] finish;
  vector[N] curr;
  array[N] int<lower=1, upper=L> ll;

}
parameters {
  array[L] vector[K] beta;
  array[L-1] real<lower=0> gamma;
  array[L] real<lower=0> sigma;
}
model {
  // priors
  for (l in 1:L) {
    beta[l] ~ normal(0, 1);
    sigma[l] ~ cauchy(0, 1); 
  }
    for (l in 1:L-1) {
    gamma[l] ~ normal(0, 1);
  }
  // model
  for (n in 1:N) {
    if (ll[n] == 1) {
      finish[n] ~ normal(feats[n] * beta[ll[n]], sigma[ll[n]]);
    } else {
      finish[n] ~ normal(feats[n] * beta[ll[n]] + curr[n] * gamma[ll[n]-1], sigma[ll[n]]);
    }
  }
}
generated quantities {
  vector[N] log_lik;
  for (n in 1:N) {
    if (ll[n] == 1) {
      log_lik[n] = normal_lpdf(finish[n] | feats[n] * beta[ll[n]], sigma[ll[n]]);
    } else {
      log_lik[n] = normal_lpdf(finish[n] | feats[n] * beta[ll[n]] + curr[n] * gamma[ll[n]-1], sigma[ll[n]]);
    }
  }
}