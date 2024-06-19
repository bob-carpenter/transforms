vector inv_alr_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  real r = log1p_exp(log_sum_exp(y));
  vector[N] x = append_row(exp(y - r), exp(-r));
  target += y;
  target += -N * r;
  return x;
}

vector inv_alr_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  real r = log1p_exp(log_sum_exp(y));
  vector[N] log_x = append_row(y - r, -r);
  target += -r;
  return log_x;
}
