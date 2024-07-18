vector inv_ilr_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] ns = linspaced_vector(N - 1, 1, N - 1);
  vector[N - 1] w = y ./ sqrt(ns .* (ns + 1));
  vector[N] z = append_row(reverse(cumulative_sum(reverse(w))), 0) - append_row(0, ns .* w);
  real r = log_sum_exp(z);
  vector[N] x = exp(z - r);
  target += 0.5 * log(N);
  target += sum(z) - N * r;
  return x;
}

vector inv_ilr_log_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] ns = linspaced_vector(N - 1, 1, N - 1);
  vector[N - 1] w = y ./ sqrt(ns .* (ns + 1));
  vector[N] z = append_row(reverse(cumulative_sum(reverse(w))), 0) - append_row(0, ns .* w);
  real r = log_sum_exp(z);
  vector[N] log_x = z - r;
  target += 0.5 * log(N);
  target += log_x[N];
  return log_x;
}
