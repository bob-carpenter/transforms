functions {
  real target_density_lp(vector x, vector alpha){
    return dirichlet_lpdf(x | alpha);
  }
}
