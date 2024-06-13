data {
  int<lower=1> N;
  vector<lower=0>[N] alpha;
}
model {
  target += dirichlet_lpdf(x | alpha);
}
