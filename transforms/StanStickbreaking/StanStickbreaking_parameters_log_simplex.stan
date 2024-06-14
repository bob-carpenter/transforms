parameters {
  simplex[N] x;
}
transformed parameters {
  vector<upper=0>[N] log_x = simplex_to_log_simplex_transform_lp(x);
}
