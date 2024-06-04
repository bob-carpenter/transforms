data {
  int<lower=1> N;
  vector<lower=0>[N] alpha;
}
model {
  target += target_density_lp(x, alpha);
}
