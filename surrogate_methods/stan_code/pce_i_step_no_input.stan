#include pce_helpers.stan

data {
  int<lower=1> N_real;      // Number of measurements per w_real
  int<lower=1> M;               // dimension of input variable
  vector[N_real] y_real;     // Output exp (measured) variables
  int<lower=1> d;               // Degree of polynomial

  matrix[d+1, d+1] l_poly_coeffs; // coefficients of legendrepolynomials
  int<lower=1> N_comb; // Number of selected polynomials
  array[N_comb, M] int comb; // polynomial selection
  real c_0;                                      // coefficient for constant "polynomial"
  vector[N_comb] c;                                   // coefficients of non-constant polynomials
  real sigma_approx; // sigma from surrogate training
  real sigma_real; // sigma for measurement noise

  vector[M] w_lower_bound;
  vector[M] w_upper_bound;
}

parameters {
  vector[M] w_real;  // estimated inputs for real (measured) variables from pce
}

model {
  target += normal_lpdf(w_real | 0, 0.5);
  matrix[1, M] input_real;
  input_real[1, 1:M] = to_row_vector(w_real);
  vector[1] mu_pred_pce_real = c_0 + get_PCE(input_real, d,
          l_poly_coeffs, comb, N_comb)*c;

  target += normal_lpdf(y_real | mu_pred_pce_real[1], sqrt(square(sigma_real)+square(sigma_approx)));
}