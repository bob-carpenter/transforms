functions {
    real exponential_log_qf(real logp){
        return -log1m_exp(logp);
    }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N] y;
}
transformed parameters {
  simplex[N] x;
  real<lower=0> r = 0;
  real log_det_jacobian = 0;
  {
    vector[N] z;
    real log_u;
    for (i in 1:N) {
      log_u = std_normal_lcdf(y[i]);
      z[i] = exponential_log_qf(log_u);
      r += z[i];
      log_det_jacobian += std_normal_lpdf(y[i]);
    }
    x = z / r;
  }
}
model {
  target += log_det_jacobian;
  target += target_density_lp(x, alpha);
}
