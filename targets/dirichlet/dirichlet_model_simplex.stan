model {
  target += dirichlet_lpdf(x | alpha);
}