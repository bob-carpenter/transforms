vector inv_ilr_reflector_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  real sqrtN = sqrt(N);
  real zN = sum(y) / sqrtN;
  vector[N] z = append_row(y - zN ./ (sqrtN - 1), zN);
  real r = log_sum_exp(z);
  vector[N] x = exp(z - r);
  target += 0.5 * log(N);
  target += -N * r;
  return x;
}

vector inv_ilr_reflector_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  real sqrtN = sqrt(N);
  real zN = sum(y) / sqrtN;
  vector[N] z = append_row(y - zN ./ (sqrtN - 1), zN);
  real r = log_sum_exp(z);
  vector[N] log_x = z - r;
  target += 0.5 * log(N);
  target += log_x[N];
  return log_x;
}
