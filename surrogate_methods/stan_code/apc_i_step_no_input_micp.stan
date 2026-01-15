#include pce_helpers.stan

data {
  int<lower=1> N_locs;      // Number of locations
  int<lower=1> N_times;     // Number of time steps
  int<lower=1> M;               // dimension of input variable
  array[N_times, N_locs] real y_real;     // Output exp (measured) variables
  int<lower=1> d;               // Degree of polynomial

  array[M] matrix[d+1, d+1] a_pc_coeffs;  // coefficients of arbitrary PC
  int<lower=1> N_comb; // Number of selected polynomials
  array[N_comb, M] int comb; // polynomial selection
  array[N_times, N_locs] real c_0;                                      // coefficient for constant "polynomial"
  array[N_times, N_locs] vector[N_comb] c;                                   // coefficients of non-constant polynomials
  array[N_times, N_locs] real sigma_approx; // sigma from surrogate training
  array[N_times, N_locs] real sigma_real; // sigma for measurement noise

  vector[M] w_lower_bound;
  vector[M] w_upper_bound;

  real prior_param_4_upper; // Upper bound for the 4th parameter (kub) in MICP model
}

parameters {
  real<lower=0, upper=1> ca1;
  real<lower=0, upper=1> ca2;
  real<lower=0, upper=1> rhoBiofilm;
  real<lower=0, upper=1> kub;
}

transformed parameters {
  vector[4] w_real;
  w_real[1] = 1e-10 + ca1 * (1e-7 - 1e-10);
  w_real[2] = 1e-10 + ca2 * (1e-6 - 1e-10);
  w_real[3] = 1 + rhoBiofilm * (15 - 1);
  w_real[4] = 1e-5 + kub * (prior_param_4_upper - 1e-5);
}

model {
  matrix[1, M] input_real;
  input_real[1, 1:M] = to_row_vector(w_real);
  matrix[1, N_comb] input_pce_basis = get_aPC(input_real, d, a_pc_coeffs, comb, N_comb);
  

  for (t_idx in 1:N_times) {
    for (loc_idx in 1:N_locs) {
      vector[1] mu_pred_pce_real = c_0[t_idx, loc_idx] + input_pce_basis*c[t_idx, loc_idx];
      target += normal_lpdf(y_real[t_idx, loc_idx] | mu_pred_pce_real[1], sqrt(square(sigma_real[t_idx, loc_idx])+square(sigma_approx[t_idx, loc_idx])));
    }
  }
  
}