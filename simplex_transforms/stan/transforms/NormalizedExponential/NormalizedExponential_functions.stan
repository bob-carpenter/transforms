real exponential_log_qf(real logp) {
  return -log1m_exp(logp);
}

vector normalized_exponential_simplex_constrain_lp(vector y) {
  int N = rows(y);
  vector[N] z;
  for (i in 1 : N) {
    real log_u = std_normal_lcdf(y[i]);
    z[i] = log(exponential_log_qf(log_u));
  }
  real r = log_sum_exp(z);
  vector[N] x = exp(z - r);
  target += std_normal_lpdf(y) - lgamma(N);
  return x;
}

vector normalized_exponential_log_simplex_constrain_lp(vector y) {
  int N = rows(y);
  vector[N] z;
  for (i in 1 : N) {
    real log_u = std_normal_lcdf(y[i]);
    z[i] = log(exponential_log_qf(log_u));
  }
  real r = log_sum_exp(z);
  vector[N] log_x = z - r;
  target += std_normal_lpdf(y) - lgamma(N);
  target += -sum(log_x[1 : N - 1]);
  return log_x;
}
