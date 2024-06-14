vector stickbricking_angular_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N] x;
  real log_phi, phi, u, s, c;
  real s2_prod = 1;
  real log_halfpi = log(pi()) - log2();
  int rcounter = 2 * N - 3;
  for (i in 1 : (N - 1)) {
    u = log_inv_logit(y[i]);
    log_phi = u + log_halfpi;
    phi = exp(log_phi);
    s = sin(phi);
    c = cos(phi);
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
  real log_phi, phi, log_u, log_s, log_c;
  real log_s2_prod = 0;
  real log_halfpi = log(pi()) - log2();
  int rcounter = 2 * N - 3;
  for (i in 1 : (N - 1)) {
    log_u = log_inv_logit(y[i]);
    log_phi = log_u + log_halfpi;
    phi = exp(log_phi);
    log_s = log(sin(phi));
    log_c = log(cos(phi));
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