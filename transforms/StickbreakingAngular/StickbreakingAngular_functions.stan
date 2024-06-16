vector stickbricking_angular_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] x;
  real s2_prod = 1;
  real log_halfpi = log(pi()) - log2();
  int rcounter = 2 * N - 3;
  for (i in 1 : (N - 1)) {
    real u = log_inv_logit(y[i]);
    real log_phi = u + log_halfpi;
    real phi = exp(log_phi);
    real s = sin(phi);
    real c = cos(phi);
    x[i] = s2_prod * c ^ 2;
    s2_prod *= s ^ 2;
    target += log_phi + log1m_exp(u) + rcounter * log(s) + log(c);
    rcounter -= 2;
  }
  x[N] = s2_prod;
  target += (N - 1) * log2();
  return x;
}

vector stickbricking_angular_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] log_x;
  real log_s2_prod = 0;
  real log_halfpi = log(pi()) - log2();
  int rcounter = 2 * N - 3;
  for (i in 1 : (N - 1)) {
    real log_u = log_inv_logit(y[i]);
    real log_phi = log_u + log_halfpi;
    real phi = exp(log_phi);
    real log_s = log(sin(phi));
    real log_c = log(cos(phi));
    log_x[i] = log_s2_prod + 2 * log_c;
    log_s2_prod += 2 * log_s;
    target += log_phi + log1m_exp(log_u) + rcounter * log_s + log_c;
    target += -log_x[i];
    rcounter -= 2;
  }
  log_x[N] = log_s2_prod;
  target += (N - 1) * log2();
  return log_x;
}
