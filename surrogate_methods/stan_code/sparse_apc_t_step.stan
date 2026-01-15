#include pce_helpers.stan

data {
  int<lower=1> M;  // dimension of input variable
  int<lower=1> N;  // number of input/output simulation pairs
  matrix[N, M] input;  // input variables
  vector[N] y;  // output variables

  int<lower=1> d;  // Degree of polynomials
  array[M] matrix[d+1, d+1] a_pc_coeffs;  // coefficients of arbitrary PC
  int<lower=1> N_comb;  // number of selected polynomials
  array[N_comb, M] int comb;  // polynomial selection
  // data for the R2D2 prior
  real<lower=0> R2D2_mean_R2;  // mean of the R2 prior
  real<lower=0> R2D2_prec_R2;  // precision of the R2 prior
  // concentration vector of the D2 prior
  vector<lower=0>[N_comb] R2D2_cons_D2;
}

transformed data {
  matrix[N, N_comb] input_pce = get_aPC(input, d, a_pc_coeffs, comb, N_comb);
}

parameters {
  real c_0; // coefficient for constant polynomial
  vector[N_comb] zc;  // unscaled coefficients of non-constant polynomials
  // parameters of the R2D2 prior
  real<lower=0,upper=1> R2D2_R2;
  simplex[N_comb] R2D2_phi;
  real<lower=0> sigma; // approximation error
}

transformed parameters {
  vector[N_comb] c;  // scaled coefficients of non-constant polynomials
  vector<lower=0>[N_comb] sdc;  // SDs of the coefficients
  real R2D2_tau2;  // global R2D2 scale parameter
  vector<lower=0>[N_comb] scales;  // local R2D2 scale parameters
  real lprior = 0;  // prior contributions to the log posterior
  // compute R2D2 scale parameters
  R2D2_tau2 = sigma^2 * R2D2_R2 / (1 - R2D2_R2);
  scales = scales_R2D2(R2D2_phi, R2D2_tau2);
  sdc = scales[(1):(N_comb)];
  c = zc .* sdc;  // scale coefficients
  lprior += normal_lpdf(c_0 | 0, 1);
  lprior += beta_lpdf(R2D2_R2 | R2D2_mean_R2 * R2D2_prec_R2, (1 - R2D2_mean_R2) * R2D2_prec_R2);
  lprior += normal_lpdf(sigma | 0, 0.1)
    - 1 * normal_lccdf(0 | 0, 0.1);
}

model {
  // pce likelihood
  vector[N] output_pce = c_0 + input_pce*c;
  target += normal_lpdf(y | output_pce, sigma);

  // priors
  // priors including constants
  target += lprior;
  target += std_normal_lpdf(zc);
  target += dirichlet_lpdf(R2D2_phi | R2D2_cons_D2);

}

// generated quantities {
//   matrix[N, N_comb] input_pce_ = input_pce;
//   vector[N] output_pce_ = c_0 + input_pce*c;
// }
