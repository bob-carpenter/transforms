vector stickbreaking_power_normal_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] x;
  real log_cum_prod = 0;
  for (i in 1 : (N - 1)) {
    real log_u = std_normal_lcdf(y[i]);
    real log_w = log_u / (N - i);
    real log_z = log1m_exp(log_w);
    x[i] = exp(log_cum_prod + log_z);
    log_cum_prod += log1m_exp(log_z);
  }
  x[N] = exp(log_cum_prod);
  target += std_normal_lpdf(y);
  target += -lgamma(N);
  return x;
}

vector stickbreaking_power_normal_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] log_x;
  real log_cum_prod = 0;
  for (i in 1 : (N - 1)) {
    real log_u = std_normal_lcdf(y[i]);
    real log_w = log_u / (N - i);
    real log_z = log1m_exp(log_w);
    log_x[i] = log_cum_prod + log_z;
    target += -log_x[i];
    log_cum_prod += log1m_exp(log_z);
  }
  log_x[N] = log_cum_prod;
  target += std_normal_lpdf(y);
  target += -lgamma(N);
  return log_x;
}