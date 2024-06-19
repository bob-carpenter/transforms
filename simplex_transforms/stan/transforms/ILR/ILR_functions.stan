matrix semiorthogonal_matrix(data int N) {
  matrix[N, N - 1] V;
  real inv_nrm2;
  for (n in 1 : (N - 1)) {
    inv_nrm2 = inv_sqrt(n * (n + 1));
    V[1 : n, n] = rep_vector(inv_nrm2, n);
    V[n + 1, n] = -n * inv_nrm2;
    V[(n + 2) : N, n] = rep_vector(0, N - n - 1);
  }
  return V;
}

vector inv_ilr_simplex_constrain_lp(vector y, data matrix V) {
  int N = rows(y) + 1;
  vector[N] z = V * y;
  real r = log_sum_exp(z);
  vector[N] x = exp(z - r);
  target += z;
  target += 0.5 * log(N) - N * r;
  return x;
}

vector inv_ilr_log_simplex_constrain_lp(vector y, data matrix V) {
  int N = rows(y) + 1;
  vector[N] z = V * y;
  real r = log_sum_exp(z);
  vector[N] log_x = z - r;
  target += 0.5 * log(N) + log_x[N];
  return log_x;
}
