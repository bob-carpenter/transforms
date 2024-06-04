functions {
  vector stickbreaking_logistic_simplex_constrain_lp(vector y) {
    int N = rows(y) + 1;
    vector[N] x;
    real log_zi, log_xi;
    real log_cum_prod = 0;
    for (i in 1:N - 1) {
      log_zi = log_inv_logit(y[i] - log(N - i)); // logistic_lcdf(y[i] | log(N - i), 1)
      log_xi = log_cum_prod + log_zi;
      x[i] = exp(log_xi);
      log_cum_prod += log1m_exp(log_zi);
      target += log_xi;
    }
    x[N] = exp(log_cum_prod);
    target += log_cum_prod;
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
  simplex[N] x = stickbreaking_logistic_simplex_constrain_lp(y);
}
model {
  target += target_density_lp(x, alpha);
}
