vector stickbricking_angular_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] log_u = log_inv_logit(y);
  vector[N - 1] log_phi = log_u + (log(pi()) - log2());
  vector[N - 1] phi = exp(log_phi);
  vector[N - 1] log_s = log(sin(phi));
  vector[N - 1] log_c = log(cos(phi));
  vector[N] log_s2_prod = append_row(0, 2 * cumulative_sum(log_s));
  vector[N] x = exp(log_s2_prod + append_row(2 * log_c, 0));
  target += (N - 1) * log2();
  target += log1m_exp(log_u);
  target += log_s2_prod[2 : N - 1];
  target += log_c;
  target += log_s;
  target += log_phi;
  return x;
}

vector stickbricking_angular_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] log_u = log_inv_logit(y);
  vector[N - 1] log_phi = log_u + (log(pi()) - log2());
  vector[N - 1] phi = exp(log_phi);
  vector[N - 1] log_s = log(sin(phi));
  vector[N - 1] log_c = log(cos(phi));
  vector[N] log_s2_prod = append_row(0, 2 * cumulative_sum(log_s));
  vector[N] log_x = log_s2_prod + append_row(2 * log_c, 0);
  target += (N - 1) * log2();
  target += log1m_exp(log_u);
  target += -sum(log_c);
  target += log_s;
  target += log_phi;
  return log_x;
}
