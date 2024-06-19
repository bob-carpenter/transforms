model {
  target += multi_logit_normal_cholesky_lpdf(x | mu, L_Sigma);
}
