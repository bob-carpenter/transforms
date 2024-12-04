vector inv_ilr_householder_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  real sqrt_N = sqrt(N);
  real norm_val = sum(y) / (1 + sqrt_N);
  real r = log_sum_exp(append_row(y, -norm_val));
  vector[N] x = exp(append_row(y, -norm_val) - r);
  target += 0.5 * log(N);
  target += sqrt_N * norm_val - N * r;
  return x;
} 

vector inv_ilr_householder_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  real sqrt_N = sqrt(N);
  real norm_val = sum(y) / (1 + sqrt_N);
  real r = log_sum_exp(append_row(y, -norm_val));
  vector[N] log_x = append_row(y, -norm_val) - r;
  target += 0.5 * log(N);
  target += log_x[N];
  return log_x;
}