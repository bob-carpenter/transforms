functions {
  vector stickbreaking_power_logistic_simplex_constrain_lp(vector y) {
    int N = rows(y) + 1;
    vector[N] x;
    real log_u, log_w, log_z;
    real log_cum_prod = 0;
    for (i in 1:(N-1)) {
      log_u = log_inv_logit(y[i]); // logistic_lcdf(y[i] | 0, 1);
      log_w = log_u / (N - i);
      log_z = log1m_exp(log_w);
      x[i] = exp(log_cum_prod + log_z);
      target += 2 * log_u - y[i]; // logistic_lupdf(y[i] | 0, 1);
      log_cum_prod += log1m_exp(log_z);
    }
    x[N] = exp(log_cum_prod);
    target += -lgamma(N);
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
  simplex[N] x = stickbreaking_power_logistic_simplex_constrain_lp(y);
}
model {
  target += target_density_lp(x, alpha);
}
