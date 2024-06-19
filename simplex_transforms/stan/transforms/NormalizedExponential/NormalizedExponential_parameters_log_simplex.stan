parameters {
  vector[N] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = normalized_exponential_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
