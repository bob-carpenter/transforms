vector inv_alr_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  real r = log1p_exp(log_sum_exp(y));
  vector[N] x;
  x[1 : N - 1] = exp(y - r);
  x[N] = exp(-r);
  target += sum(y) - N * r;
  return x;
}

vector inv_alr_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  real r = log1p_exp(log_sum_exp(y));
  vector[N] log_x;
  log_x[1 : N - 1] = y - r;
  log_x[N] = -r;
  target += -r;
  return log_x;
}
