vector expanded_softmax_simplex_constrain_lp(vector y) {
  int N = rows(y);
  real r = log_sum_exp(y);
  vector[N] x = exp(y - r);
  target += std_normal_lpdf(r - log(N));
  target += sum(y) - N * r;
  return x;
}

vector expanded_softmax_log_simplex_constrain_lp(vector y) {
  int N = rows(y);
  real r = log_sum_exp(y);
  vector[N] log_x = y - r;
  target += std_normal_lpdf(r - log(N));
  target += log_x[N];
  return log_x;
}
