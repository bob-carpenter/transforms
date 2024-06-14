parameters {
  vector[N - 1] y;
}
transformed parameters {
  vector<upper=0>[N] log_x = stickbreaking_logistic_log_simplex_constrain_lp(y);
  simplex[N] x = exp(log_x);
}
