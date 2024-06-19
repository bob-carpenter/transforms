/**
 * Return the Dirichlet density for the specified log simplex.
 *
 * @param theta a vector on the log simplex (N rows)
 * @param alpha prior counts plus one (N rows)
 */
real log_dirichlet_lpdf(vector log_theta, vector alpha) {
  int N = rows(log_theta);
  if (N != rows(alpha))
    reject("Input must contain same number of elements as alpha");
  return dot_product(alpha, log_theta) - log_theta[N]
         + lgamma(sum(alpha)) - sum(lgamma(alpha));
}
