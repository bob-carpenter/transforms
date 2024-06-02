functions {
  matrix semiorthogonal_matrix(int N) {
    matrix[N, N - 1] V;
    real inv_nrm2;
    for (n in 1:(N - 1)) {
      inv_nrm2 = inv_sqrt(n * (n + 1));
      V[1:n, n] = rep_vector(inv_nrm2, n);
      V[n + 1, n] = -n * inv_nrm2;
      V[(n + 2):N, n] = rep_vector(0, N - n - 1);
    }
    return V;
  }

  vector inv_ilr_simplex_constrain_lp(vector y, matrix V) {
    int N = rows(y) + 1;
    vector[N] z = V * y;
    real logr = log_sum_exp(z);
    vector[N] x = exp(z - logr);
    target += sum(z) - N * logr + 0.5 * log(N);
    return x;
  }
}
data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
transformed data {
  matrix[N, N - 1] V = semiorthogonal_matrix(N);
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x = inv_ilr_simplex_constrain_lp(y, V);
}
model {
  target += target_density_lp(x, alpha);
}
