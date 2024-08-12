vector exponential_simplex_constrain_lp(vector y) {
  int N = rows(y);
  vector[N] z = y - log1p_exp(-y);
  real r = log_sum_exp(x_pos);
  vector[N] x = exp(x_pos - r);
  target += std_normal_lpdf(r - log(N));
  target += sum(y);
  return x;
}

vector exponential_log_simplex_constrain_lp(vector y) {
  int N = rows(y);
  vector[N] x = y - log1p_exp(-y);
  real r = log_sum_exp(x);
  vector[N] log_x = x - r;
  target += std_normal_lpdf(r - log(N));
  target += log_x[N] - N * r;
  return log_x;
}
