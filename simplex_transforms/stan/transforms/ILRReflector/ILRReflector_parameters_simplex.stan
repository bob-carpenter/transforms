parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x = inv_ilr_reflector_simplex_constrain_lp(y);
}
