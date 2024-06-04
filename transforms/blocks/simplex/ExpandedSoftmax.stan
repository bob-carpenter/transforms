functions{
  vector expanded_softmax_simplex_constrain_lp(vector y) {
    int N = rows(y);
    real r = log_sum_exp(y);
    vector[N] x = exp(y - r);
    target += sum(y) - N * r;  // target += log(prod(x))
    target += std_normal_lpdf(r - log(N));
    return x;
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
  simplex[N] x = expanded_softmax_simplex_constrain_lp(y);
}
model {
  target += target_density_lp(x, alpha);
}
