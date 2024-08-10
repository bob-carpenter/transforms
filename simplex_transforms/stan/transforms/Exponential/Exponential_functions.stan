vector exponential_simplex_constrain_lp(vector y) {
  int N = rows(y);
  vector[N] x_pos = y - log1p_exp(-y);
  real r = log_sum_exp(x_pos);
  vector[N] x = exp(x_pos - r);
  target += x_pos;
  target += std_normal_lpdf(r - log(N));
  target += sum(y) - N * r;
  return x;
}

vector exponential_log_simplex_constrain_lp(vector y) {
  int N = rows(y);
  vector[N] x = y - log1p_exp(-y);
  real r = log_sum_exp(x);
  vector[N] log_x = x - r;
  target += x;
  target += std_normal_lpdf(r - log(N));
  target += log_x[N];
  return log_x;
}
