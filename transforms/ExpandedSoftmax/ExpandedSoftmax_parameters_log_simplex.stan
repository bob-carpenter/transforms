parameters {
  vector[N] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = expanded_softmax_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
