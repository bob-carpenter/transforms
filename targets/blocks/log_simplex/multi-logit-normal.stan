data {
  int<lower=1> N;
  vector[N-1] mu;
  matrix[N-1, N-1] L_Sigma;
}
model {
  target += log_multi_logit_normal_cholesky_lpdf(log_x | mu, L_Sigma);
}
