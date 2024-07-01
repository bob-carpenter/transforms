real none_lpdf(vector theta) {
  return 0;
}

real log_none_lpdf(vector log_theta) {
  int N = rows(log_theta);
  return sum(log_theta[1 : N - 1]);
}
