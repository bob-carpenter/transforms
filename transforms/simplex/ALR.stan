functions {
  vector inv_alr_simplex_constrain_lp(vector y){
    int N = rows(y) + 1;
    real r = log1p_exp(log_sum_exp(y));
    vector[N] x;
    x[1:N - 1] = exp(y - r);
    x[N] = exp(-r);
    target += sum(y) - N * r;
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
  simplex[N] x = inv_alr_simplex_constrain_lp(y);
}
model {
  target += target_density_lp(x, alpha);
}
