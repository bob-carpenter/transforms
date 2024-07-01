vector std_normal_lcdf_vector(vector x) {
  int N = rows(x);
  vector[N] y;
  for (n in 1 : N)
    y[n] = std_normal_lcdf(x[n]);
  return y;
}

vector stickbreaking_power_normal_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] log_u = std_normal_lcdf_vector(y);
  vector[N - 1] log_w = log_u ./ reverse(linspaced_vector(N - 1, 1, N - 1));
  vector[N] x = exp(append_row(log1m_exp(log_w), 0) + append_row(0, cumulative_sum(log_w)));
  target += std_normal_lpdf(y);
  target += -lgamma(N);
  return x;
}

vector stickbreaking_power_normal_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] log_u = std_normal_lcdf_vector(y);
  vector[N - 1] log_w = log_u ./ reverse(linspaced_vector(N - 1, 1, N - 1));
  vector[N] log_x = append_row(log1m_exp(log_w), 0) + append_row(0, cumulative_sum(log_w));
  target += std_normal_lpdf(y);
  target += -lgamma(N);
  target += -sum(log_x[1 : N - 1]);
  return log_x;
}
