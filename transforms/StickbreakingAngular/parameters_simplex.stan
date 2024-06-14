parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x = stickbricking_angular_simplex_constrain_lp(y);
}
