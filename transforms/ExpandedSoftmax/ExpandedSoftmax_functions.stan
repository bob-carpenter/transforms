vector expanded_softmax_simplex_constrain_lp(vector y) {
  int N = rows(y);
  real r = log_sum_exp(y);
  vector[N] x = exp(y - r);
  target += sum(y) - N * r; // target += log(prod(x))
  target += std_normal_lpdf(r - log(N));
  return x;
}

vector expanded_softmax_log_simplex_constrain_lp(vector y) {
  int N = rows(y);
  real r = log_sum_exp(y);
  vector[N] log_x = y - r;
  target += log_x[N];
  target += std_normal_lpdf(r - log(N));
  return log_x;
}
