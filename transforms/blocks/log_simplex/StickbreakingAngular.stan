functions {
  vector stickbricking_angular_log_simplex_constrain_lp(vector y) {
    int N = rows(y) + 1;
    vector[N] log_x;
    real log_phi, phi, log_u, log_s, log_c;
    real log_s2_prod = 0;
    real log_halfpi = log(pi()) - log2();
    int rcounter = 2 * N - 3;
    for (i in 1:(N-1)) {
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
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = stickbricking_angular_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
model {
  target += target_density_lp(log_x, alpha);
}
