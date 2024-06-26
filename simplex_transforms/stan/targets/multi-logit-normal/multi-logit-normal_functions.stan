/**
 * Return the multivariate logistic normal density for the specified simplex.
 *
 * See: https://en.wikipedia.org/wiki/Logit-normal_distribution#Multivariate_generalization
 *
 * @param theta a vector on the simplex (N rows)
 * @param mu location of normal (N-1 rows)
 * @param L_Sigma Cholesky factor of covariance (N-1 rows, N-1 cols)
 */
real multi_logit_normal_cholesky_lpdf(vector theta, vector mu, matrix L_Sigma) {
  int N = rows(theta);
  vector[N] log_theta = log(theta);
  return multi_normal_cholesky_lpdf(log_theta[1 : N - 1] - log_theta[N] | mu, L_Sigma)
         - sum(log_theta);
}

/**
 * Return the multivariate logistic normal density for the specified log simplex.
 *
 * See: https://en.wikipedia.org/wiki/Logit-normal_distribution#Multivariate_generalization
 *
 * @param theta a vector on the log simplex (N rows)
 * @param mu location of normal (N-1 rows)
 * @param L_Sigma Cholesky factor of covariance (N-1 rows, N-1 cols)
 */
real log_multi_logit_normal_cholesky_lpdf(vector log_theta, vector mu,
                                          matrix L_Sigma) {
  int N = rows(log_theta);
  return multi_normal_cholesky_lpdf(log_theta[1 : N - 1] - log_theta[N] | mu, L_Sigma)
         - log_theta[N];
}
