functions {
  vector stickbreaking_power_normal_log_simplex_constrain_lp(vector y) {
    int N = rows(y) + 1;
    vector[N] log_x;
    real log_u, log_w, log_z;
    real log_cum_prod = 0;
    for (i in 1:(N-1)) {
      log_u = std_normal_lcdf(y[i] |);
      log_w = log_u / (N - i);
      log_z = log1m_exp(log_w);
      log_x[i] = log_cum_prod + log_z;
      target += std_normal_lpdf(y[i] |);
      target += -log_x[i];
      log_cum_prod += log1m_exp(log_z);
    }
    log_x[N] = log_cum_prod;
    target += -lgamma(N);
    return log_x;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = stickbreaking_power_normal_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
model {
  target += target_density_lp(log_x, alpha);
}
