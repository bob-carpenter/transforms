functions {
  real exponential_log_qf(real logp){
    return -log1m_exp(logp);
  }
  vector normalized_exponential_log_simplex_constrain_lp(vector y) {
    int N = rows(y);
    vector[N] z;
    real log_u;
    for (i in 1:N) {
      log_u = std_normal_lcdf(y[i]);
      z[i] = log(exponential_log_qf(log_u));
    }
    real log_r = log_sum_exp(z);
    vector[N] log_x = z - log_r;
    target += -log_x[1:N - 1];
    target += std_normal_lpdf(y) - lgamma(N);
    return log_x;
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
  vector<upper=0>[N] log_x = normalized_exponential_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
model {
  target += target_density_lp(log_x, alpha);
}
