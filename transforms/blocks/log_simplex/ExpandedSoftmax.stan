functions{
  vector expanded_softmax_log_simplex_constrain_lp(vector y) {
    int N = rows(y);
    real r = log_sum_exp(y);
    vector[N] log_x = y - r;
    target += log_x[N];
    target += std_normal_lpdf(r - log(N));
    return log_x;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = expanded_softmax_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
model {
  target += target_density_lp(log_x, alpha);
}
