data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  simplex[N] x;
}
transformed parameters {
  vector<upper=0>[N] log_x = log(x);
}
model {
  for i in 1:(N - 1)
    target += -log_x[i];
  target += target_density_lp(log_x, alpha);
}
