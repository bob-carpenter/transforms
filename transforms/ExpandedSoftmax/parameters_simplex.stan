parameters {
  vector[N] y;
}
transformed parameters {
  simplex[N] x = expanded_softmax_simplex_constrain_lp(y);
}