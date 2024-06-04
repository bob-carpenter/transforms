functions {
  real log_dirichlet_lpdf(vector log_theta, vector alpha) {
    int N = rows(log_theta);
    if (N != rows(alpha)) reject("Input must contain same number of elements as alpha");
    real lp = dot_product(alpha, log_theta) - log_theta[N];
    lp += sum(lgamma(alpha)) - lgamma(sum(alpha));
    return lp;
  }
  real target_density_lp(vector log_x, vector alpha){
    return log_dirichlet_lpdf(log_x | alpha);
  }
}
