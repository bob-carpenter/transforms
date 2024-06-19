parameters {
  vector[N] y;
}
transformed parameters {
  simplex[N] x = normalized_exponential_simplex_constrain_lp(y);
}
