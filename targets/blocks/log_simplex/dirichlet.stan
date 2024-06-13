data {
  int<lower=1> N;
  vector<lower=0>[N] alpha;
}
model {
  target += log_dirichlet_lpdf(log_x | alpha);
}
