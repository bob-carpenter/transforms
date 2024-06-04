functions {
  vector stickbreaking_normal_simplex_constrain_lp(vector y) {
    int N = rows(y) + 1;
    vector[N] x;
    real log_zi, log_xi, wi;
    real log_cum_prod = 0;
    for (i in 1:N - 1) {
      wi = y[i] - log(N - i) / 2;
      log_zi = std_normal_lcdf(wi);
      log_xi = log_cum_prod + log_zi;
      x[i] = exp(log_xi);
      target += std_normal_lpdf(wi) + log_cum_prod;
      log_cum_prod += log1m_exp(log_zi);
    }
    x[N] = exp(log_cum_prod);
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
  simplex[N] x = stickbreaking_normal_simplex_constrain_lp(y);
}
model {
  target += target_density_lp(x, alpha);
}
