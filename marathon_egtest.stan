data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=1> L;
  matrix[N, K] feats;
  vector[N] finish;
  array[N] int<lower=1, upper=L> ll;
}
parameters {
  array[L] vector[K] beta;
  array[L] real<lower=0> sigma;
}
model {
  // priors
  for (l in 1:L) {
    beta[l] ~ normal(0, 1);
    sigma[l] ~ cauchy(0, 1); 
  }
  // model
  for (n in 1:N) {
  // Dr. Gerber thinks something like this might work (check)
    if (L == 1) {
        // this is for the 5K split; don't estimate the beta for curr_pace (leave 0)
        finish[n] ~ normal(feats[n,-3] * beta[ll[n],-3], sigma[ll[n]]);
    else {
        // for all other splits, do as you used to
        finish[n] ~ normal(feats[n] * beta[ll[n]], sigma[ll[n]]);
    }
  }
}
