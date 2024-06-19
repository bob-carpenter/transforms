vector stickbreaking_power_logistic_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] log_u = log_inv_logit(y); // logistic_lcdf(y[i] | 0, 1)
  vector[N - 1] log_w = log_u ./ reverse(linspaced_vector(N - 1, 1, N - 1));
  vector[N] x = exp(append_row(log1m_exp(log_w), 0) + append_row(0, cumulative_sum(log_w)));
  // logistic_lpdf(y[i] | 0, 1)
  target += 2 * sum(log_u) - sum(y);
  target += -lgamma(N);
  return x;
}

vector stickbreaking_power_logistic_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] log_u = log_inv_logit(y); // logistic_lcdf(y[i] | 0, 1)
  vector[N - 1] log_w = log_u ./ reverse(linspaced_vector(N - 1, 1, N - 1));
  vector[N] log_x = append_row(log1m_exp(log_w), 0) + append_row(0, cumulative_sum(log_w));
  // logistic_lpdf(y[i] | 0, 1)
  target += 2 * sum(log_u) - sum(y);
  target += -lgamma(N);
  target += -log_x[1 : N - 1];
  return log_x;
}
