data {
  int<lower=1> N;
  vector[N - 1] mu;
  matrix[N - 1, N - 1] L_Sigma;
}