parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x = stickbreaking_power_normal_simplex_constrain_lp(y);
}
