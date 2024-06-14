transformed data {
  matrix[N, N - 1] V = semiorthogonal_matrix(N);
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x = inv_ilr_simplex_constrain_lp(y, V);
}
