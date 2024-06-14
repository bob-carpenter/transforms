vector simplex_to_log_simplex_transform_lp(vector x) {
  int N = rows(x);
  vector[N] log_x = log(x);
  target += -log_x[1 : N - 1];
  return log_x;
}