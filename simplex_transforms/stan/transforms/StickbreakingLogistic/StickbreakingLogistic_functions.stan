vector stickbreaking_logistic_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  // vectorized log_z[i] = logistic_lcdf(y[i] | log(N - i), 1)
  vector[N - 1] log_z = log_inv_logit(y - log(reverse(linspaced_vector(N - 1, 1, N - 1))));
  vector[N] log_cum_prod = append_row(0, cumulative_sum(log1m_exp(log_z)));
  vector[N] x = exp(append_row(log_z, 0) + log_cum_prod);
  target += sum(log_cum_prod) + sum(log_z);
  return x;
}

vector stickbreaking_logistic_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  // vectorized log_z[i] = logistic_lcdf(y[i] | log(N - i), 1)
  vector[N - 1] log_z = log_inv_logit(y - log(reverse(linspaced_vector(N - 1, 1, N - 1))));
  vector[N] log_x = append_row(log_z, 0) + append_row(0, cumulative_sum(log1m_exp(log_z)));
  target += log_x[N];
  return log_x;
}
