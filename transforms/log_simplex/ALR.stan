functions {
  vector inv_alr_log_simplex_constrain_lp(vector y){
    int N = rows(y) + 1;
    real r = log1p_exp(log_sum_exp(y));
    vector[N] log_x;
    log_x[1:N - 1] = y - r;
    log_x[N] = -r;
    target += -r;
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
  vector<upper=0>[N] log_x = inv_alr_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
model {
  target += target_density_lp(log_x, alpha);
}
