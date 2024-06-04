functions {
  vector stickbricking_angular_simplex_constrain_lp(vector y) {
    int N = rows(y) + 1;
    vector[N] x;
    real log_phi, phi, u, s, c;
    real s2_prod = 1;
    real log_halfpi = log(pi()) - log2();
    int rcounter = 2 * N - 3;
    for (i in 1:(N-1)) {
      u = log_inv_logit(y[i]);
      log_phi = u + log_halfpi;
      phi = exp(log_phi);
      s = sin(phi);
      c = cos(phi);
      x[i] = s2_prod * c^2;
      s2_prod *= s^2;
      target += log_phi + log1m_exp(u) + rcounter * log(s) + log(c);
      rcounter -= 2;
    }
    x[N] = s2_prod;
    target += (N - 1) * log2();
    return x;
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
  simplex[N] x = stickbricking_angular_simplex_constrain_lp(y);
}
model {
  target += target_density_lp(x, alpha);
}
