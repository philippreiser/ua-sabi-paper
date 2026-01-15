from typing import Dict, Type
import numpy as np
import os
import cmdstanpy as csp
import arviz as az

from .pce_utils import get_pce_vars, get_pce, poly_idx
from utils import scale_to_1


class BayesianPCEBase:
    def __init__(self, polynomial_degree: int, log2_n_training_points: int, n_samples: int, n_warmup: int, n_chains: int, prop_sigma_approx: bool, save_path: str, model: Type):
        """
        Initialize the Bayesian PCE, a class to train a Bayesian PCE using MCMC via cmdstan.

        Args:
            polynomial_degree (int): Degree of the polynomial chaos expansion.
            log2_n_training_points (int): Log base 2 of the number of training points.
            n_samples (int): Number of samples for the MCMC.
            n_warmup (int): Number of warmup iterations for the MCMC.
            n_chains (int): Number of MCMC chains.
            prop_sigma_approx (bool): If True, the approximation error is added to the output.
            save_path (str): Path to save the fitted model.
            model (Type): The model object containing the sobol_sequence method and 
            the dim_in and dim_param attributes.
        """

        self.polynomial_degree = polynomial_degree
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.n_chains = n_chains
        self.log2_n_training_points = log2_n_training_points
        self.prop_sigma_approx = prop_sigma_approx
        self.model = model
        self.n_total_samples = n_chains * n_samples
        self.poly_idx = poly_idx(
            polynomial_degree, model.dim_in + model.dim_param)
        self.c_dim = len(self.poly_idx) - 1
        self.save_path = save_path

    def get_training_data(self) -> Dict:
        """
        Prepare training data and pce coefficients for the cmdstan model.

        Returns:
            Dict: A dictionary containing the training data.
        """

        train_data = self.model.sobol_sequence(
            m=self.log2_n_training_points)
        # Extract inputs and outputs from train data, assuming
        # the output is only 1-dim -> adapt for multi-dim output
        inputs = train_data[:-1]
        output = train_data[-1]
        inputs_scaled = np.stack(self.model.scale_inputs_to_1(*inputs), axis=1)
        assert inputs_scaled.shape[1] == self.model.dim_param + self.model.dim_in, \
            "Inputs do not match the expected dimensions of the model."

        (l_poly_coeffs_mat, comb) = get_pce_vars(
            self.polynomial_degree, (self.model.dim_param + self.model.dim_in))

        training_data = {
            'M': self.model.dim_in + self.model.dim_param,
            'N': inputs_scaled.shape[0],
            'y': output,
            'input': inputs_scaled,
            'd': self.polynomial_degree,
            'l_poly_coeffs': l_poly_coeffs_mat.T,
            'N_comb': len(comb),
            'comb': comb
        }

        return training_data

    def train(self):
        """
        Train the Bayesian PCE model using the training data.
        """
        training_data = self.get_training_data()

        # Define Bayesian surrogate
        stan_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'stan_code', 'pce_t_step.stan')
        model = csp.CmdStanModel(stan_file=stan_file)

        # Fit surrogate
        fit = model.sample(data=training_data, adapt_delta=0.99, seed=100,
                           iter_warmup=self.n_warmup, iter_sampling=self.n_samples, chains=self.n_chains,
                           inits=0)
        self.stan_model = model
        self.fit = fit
        idata = az.from_cmdstanpy(self.fit)
        self.c_0_samples = np.array(idata.posterior.c_0).reshape(
            self.n_total_samples, -1)
        self.c_samples = np.array(idata.posterior.c).reshape(
            self.n_total_samples, -1)
        self.sigma_samples = np.array(
            idata.posterior.sigma).reshape(self.n_total_samples, -1)
        self.c_0_median = np.median(self.c_0_samples, axis=0)
        self.c_median = np.median(self.c_samples, axis=0)
        self.sigma_median = np.median(self.sigma_samples, axis=0)

    def _prepare_inputs(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Normalize inputs to [-1, 1] using the parameter/input ranges defined by the model.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.

        Returns:
            np.ndarray: A two-dimensional array of shape (n, m), where 'n' is the 
            number of samples and 'm' is the model's total input dimensionality 
            ('dim_in + dim_param').
        """

        if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in inputs):
            raise ValueError("All inputs must be one-dimensional np.ndarrays.")
        if len({arr.shape[0] for arr in inputs}) != 1:
            raise ValueError("All input arrays must have the same length.")
        inputs_prepared = np.column_stack(
            self.model.scale_inputs_to_1(*inputs))
        if inputs_prepared.shape[1] != (self.model.dim_in + self.model.dim_param):
            raise ValueError(
                "Inputs do not match the expected dimensions of the model.")
        return inputs_prepared

    def get_measurement_data(self, measurments: np.ndarray, inputs: np.ndarray) -> Dict:
        """
        Prepares the measurement data dictionary required for Stan inference.

        Args:
        measurements (np.ndarray): 
            1D array of observed measurement data.
        inputs (np.ndarray): 
            2D array of input values associated with the measurements. 
            If 'dim_in == 0', this is ignored.

        Returns:
            Dict: A dictionary containing the measurement data.
        """

        (l_poly_coeffs_mat, comb) = get_pce_vars(self.polynomial_degree,
                                                 (self.model.dim_param + self.model.dim_in))

        measurement_data = {
            'N_real': len(measurments),
            'M': self.model.dim_in + self.model.dim_param,
            'y_real': measurments,
            'd': self.polynomial_degree,
            'l_poly_coeffs': l_poly_coeffs_mat.T,
            'N_comb': len(comb),
            'comb': comb,
            'w_lower_bound': -1 * np.ones(self.model.dim_param),
            'w_upper_bound': np.ones(self.model.dim_param),
            'sigma_real': self.model._sigmaI
        }

        if self.model.dim_in > 0:
            measurement_data['L'] = self.model.dim_in
            measurement_data['x_real'] = inputs.reshape(-1, self.model.dim_in)
            measurement_data['w_idxs'] = np.arange(1, self.model.dim_param+1)
            measurement_data['x_idxs'] = np.arange(
                self.model.dim_param+1, self.model.dim_in+self.model.dim_param+1)

        return measurement_data

    def evaluate(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Abstract method for evaluation; must be implemented by subclasses.
        """

        raise NotImplementedError(
            "Subclasses must implement the 'evaluate' method.")

    def save(self):
        """
        Save the fitted cmdstan model.
        """


class BayesianPCE(BayesianPCEBase):
    def __init__(self, polynomial_degree: int, log2_n_training_points: int, n_samples: int, n_warmup: int, n_chains: int, prop_sigma_approx: bool, save_path: str, model: Type):
        """
        Initialize BayesianPCE.

        Args:
            polynomial_degree (int): Degree of the polynomial chaos expansion.
            log2_n_training_points (int): Log base 2 of the number of training points.
            n_samples (int): Number of samples for the MCMC.
            n_warmup (int): Number of warmup iterations for the MCMC.
            n_chains (int): Number of MCMC chains.
            prop_sigma_approx (bool): If True, the approximation error is added to the output.
            save_path (str): Path to save the fitted model.
            model (Type): The model object containing the sobol_sequence method and 
            the dim_in and dim_param attributes.
        """

        super().__init__(polynomial_degree, log2_n_training_points, n_samples,
                         n_warmup, n_chains, prop_sigma_approx, save_path, model)

    def evaluate(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the Bayesian PCE by sampling a single draw
        from the surrogate posterior.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.

        Returns:
            np.ndarray: An array of shape (n,) with a random posterior predictive draw.
        """

        inputs_prepared = self._prepare_inputs(*inputs)
        pce_basis = get_pce(*inputs_prepared.T,
                            p=self.polynomial_degree, comb=self.poly_idx)

        # Sample a random draw from the surrogate posterior
        random_idx = np.random.randint(low=0, high=(self.n_total_samples-1))
        c_0_draw = self.c_0_samples[random_idx, :]
        c_draw = self.c_samples[random_idx, :]
        sigma_draw = self.sigma_samples[random_idx, :]
        out = c_draw @ pce_basis.T + c_0_draw
        if self.prop_sigma_approx:
            out = np.random.normal(loc=out, scale=sigma_draw)
        return out

    def evaluate_full_posterior_predictive(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the Bayesian PCE by returning the
        full posterior predictive.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.

        Returns:
            np.ndarray: An array of shape (n,) with a random posterior predictive draw.
        """

        inputs_prepared = self._prepare_inputs(*inputs)
        pce_basis = get_pce(*inputs_prepared.T,
                            p=self.polynomial_degree, comb=self.poly_idx)
        n_points = inputs_prepared.shape[0]

        # Calculate PCE basis
        pce_basis = get_pce(*inputs_prepared.T, p=self.polynomial_degree)

        # Surrogate posterior mean prediction
        out_mean_samples = self.c_samples @ pce_basis.T + self.c_0_samples

        # Surrogate posterior predictive
        out_samples = np.random.normal(
            loc=out_mean_samples, scale=self.sigma_samples)
        return out_samples


class BayesianPointPCE(BayesianPCEBase):
    def __init__(self, polynomial_degree: int, log2_n_training_points: int, n_samples: int, n_warmup: int, n_chains: int, prop_sigma_approx: bool, save_path: str, model: Type):
        """
        Initialize BayesianPCE.

        Args:
            polynomial_degree (int): Degree of the polynomial chaos expansion.
            log2_n_training_points (int): Log base 2 of the number of training points.
            n_samples (int): Number of samples for the MCMC.
            n_warmup (int): Number of warmup iterations for the MCMC.
            n_chains (int): Number of MCMC chains.
            prop_sigma_approx (bool): If True, the approximation error is added to the output.
            save_path (str): Path to save the fitted model.
            model (Type): The model object containing the sobol_sequence method and 
            the dim_in and dim_param attributes.
        """

        super().__init__(polynomial_degree, log2_n_training_points, n_samples,
                         n_warmup, n_chains, prop_sigma_approx, save_path, model)

    def evaluate(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the Bayesian PCE by calculating the median of the posterior
        of the surrogate coefficients.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.
            prop_sigma_approx (bool): If True, an additional error_approx
            is added to the output

        Returns:
            np.ndarray: An array of shape (n,) with a random posterior predictive draw.
        """

        inputs_prepared = self._prepare_inputs(*inputs)
        pce_basis = get_pce(*inputs_prepared.T,
                            p=self.polynomial_degree, comb=self.poly_idx)
        out = self.c_median @ pce_basis.T + self.c_0_median
        if self.prop_sigma_approx:
            out = np.random.normal(loc=out, scale=self.sigma_median)
        return out
