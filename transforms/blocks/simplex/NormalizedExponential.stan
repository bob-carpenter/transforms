functions {
  real exponential_log_qf(real logp){
    return -log1m_exp(logp);
  }
  vector normalized_exponential_simplex_constrain_lp(vector y) {
    int N = rows(y);
    vector[N] z;
    real log_u;
    for (i in 1:N) {
      log_u = std_normal_lcdf(y[i]);
      z[i] = log(exponential_log_qf(log_u));
    }
    real r = log_sum_exp(z);
    vector[N] x = exp(z - r);
    target += std_normal_lpdf(y) - lgamma(N);
    return x;
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
  simplex[N] x = normalized_exponential_simplex_constrain_lp(y);
}
model {
  target += target_density_lp(x, alpha);
}
