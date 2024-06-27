vector std_normal_lcdf_vector(vector x) {
  int N = rows(x);
  vector[N] y;
  for (n in 1 : N)
    y[n] = std_normal_lcdf(x[n]);
  return y;
}

vector stickbreaking_normal_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  // vectorized log_z[i] = normal_lcdf(y[i] | log(N - i) / 2, 1)
  vector[N - 1] w = y - log(reverse(linspaced_vector(N - 1, 1, N - 1))) / 2;
  vector[N - 1] log_z = std_normal_lcdf_vector(w);
  vector[N] log_cum_prod = append_row(0, cumulative_sum(log1m_exp(log_z)));
  vector[N] x = exp(append_row(log_z, 0) + log_cum_prod);
  target += std_normal_lpdf(w);
  target += sum(log_cum_prod[2 : N - 1]);
  return x;
}

vector stickbreaking_normal_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  // vectorized log_z[i] = normal_lcdf(y[i] | log(N - i) / 2, 1)
  vector[N - 1] w = y - log(reverse(linspaced_vector(N - 1, 1, N - 1))) / 2;
  vector[N - 1] log_z = std_normal_lcdf_vector(w);
  vector[N] log_x = append_row(log_z, 0) + append_row(0, cumulative_sum(log1m_exp(log_z)));
  target += std_normal_lpdf(w);
  target += -sum(log_z);
  return log_x;
}
