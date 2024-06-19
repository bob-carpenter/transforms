parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x = stickbreaking_normal_simplex_constrain_lp(y);
}
