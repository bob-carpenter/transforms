vector stickbreaking_normal_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] x;
  real log_cum_prod = 0;
  for (i in 1 : N - 1) {
    real wi = y[i] - log(N - i) / 2;
    real log_zi = std_normal_lcdf(wi);
    real log_xi = log_cum_prod + log_zi;
    x[i] = exp(log_xi);
    target += std_normal_lpdf(wi) + log_cum_prod;
    log_cum_prod += log1m_exp(log_zi);
  }
  x[N] = exp(log_cum_prod);
  return x;
}

vector stickbreaking_normal_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] log_x;
  real log_cum_prod = 0;
  for (i in 1 : N - 1) {
    real wi = y[i] - log(N - i) / 2;
    real log_zi = std_normal_lcdf(wi);
    log_x[i] = log_cum_prod + log_zi;
    target += std_normal_lpdf(wi) - log_zi;
    log_cum_prod += log1m_exp(log_zi);
  }
  log_x[N] = log_cum_prod;
  return log_x;
}
