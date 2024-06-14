model {
  target += log_dirichlet_lpdf(log_x | alpha);
}
