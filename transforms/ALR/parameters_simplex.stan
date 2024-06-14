parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x = inv_alr_simplex_constrain_lp(y);
}