parameters {
  vector[N] y;
}
transformed parameters {
  simplex[N] x = exponential_simplex_constrain_lp(y);
}
