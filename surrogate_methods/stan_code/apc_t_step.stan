#include pce_helpers.stan

data {
  int<lower=1> M;                     // dimension of input variable
  int<lower=1> N;                     // Number of input/output simulation pairs
  matrix[N, M] input;             // input variables
  vector[N] y;                    // Output variables

  int<lower=1> d;                     // Degree of polynomials
  array[M] matrix[d+1, d+1] a_pc_coeffs;     // coefficients of arbitrary PC
  int<lower=1> N_comb;                // Number of selected polynomials
  array[N_comb, M] int comb;          // polynomial selection
}

transformed data {
  matrix[N, N_comb] input_pce = get_aPC(input, d, a_pc_coeffs, comb, N_comb);
}

parameters {
  real c_0;                               // coefficient for constant "polynomial"
  vector[N_comb] c;                       // coefficients of non-constant polynomials
  real<lower=0> sigma;                    // sigma for both simulation and real data
}

model {
  vector[N] output_pce = c_0 + input_pce*c;

  // Prior model
  c_0 ~ normal(0, 5);
  c ~ normal(0, 5);
  sigma ~ normal(0, 0.5);

  // Observational model
  target += normal_lpdf(y | output_pce, sigma);
}

// generated quantities {
//   matrix[N, N_comb] input_pce_ = input_pce;
//   vector[N] output_pce_ = c_0 + input_pce*c;
// }
