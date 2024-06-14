vector stickbreaking_logistic_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] x;
  real log_zi, log_xi;
  real log_cum_prod = 0;
  for (i in 1 : N - 1) {
    log_zi = log_inv_logit(y[i] - log(N - i)); // logistic_lcdf(y[i] | log(N - i), 1)
    log_xi = log_cum_prod + log_zi;
    x[i] = exp(log_xi);
    log_cum_prod += log1m_exp(log_zi);
    target += log_xi;
  }
  x[N] = exp(log_cum_prod);
  target += log_cum_prod;
  return x;
}

vector stickbreaking_logistic_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] log_x;
  real log_z;
  real log_cum_prod = 0;
  for (i in 1 : (N - 1)) {
    log_z = log_inv_logit(y[i] - log(N - i)); // logistic_lcdf(y[i] | log(N - i), 1)
    log_x[i] = log_cum_prod + log_z;
    log_cum_prod += log1m_exp(log_z);
  }
  log_x[N] = log_cum_prod;
  target += log_cum_prod;
  return log_x;
}