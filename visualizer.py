from typing import Dict, Any, Type, List, Tuple
import os
import numpy as np
import arviz as az
import scipy.stats
import json
import glob
import math

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import colorsys

import utils as u
import bayesflow as bf
from bayesflow.utils import make_quadratic
from bayesflow.utils.ecdf import simultaneous_ecdf_bands
from bayesflow.utils.ecdf.ranks import fractional_ranks


class Visualizer:
    """
    Manages data visualization setup, including directory creation for saving figures.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the Visualizer, setting up directories for creating and saving visualizations based on the configuration.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration settings, including paths and dataset details.
        """

        self._config = config
        self.figsize = (8, 4)
        self.rcParams = {
            "figure.figsize": self.figsize,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "lines.markersize": 10,
            "lines.linewidth": 2,
            "axes.grid": True,
            "grid.alpha": 0.7,
            "grid.linestyle": "-",
            }
        self.surrogate_color = "#1dc71d"
        self.simulation_color = "#094909"
        self.measurement_color = "orange"

        os.makedirs(os.path.join(config.output_path, "figures"), exist_ok=True)

    def _generate_shades(self, base_color, num_colors) -> List[str]:
        """
        Generate a color palette by adjusting the brightness of a given base color.

        Args:
            base_color (str): Hex code of the base color.
            num_colors (int): Number of colors to generate.

        Returns:
            list: A list of color hex codes.
        """

        # if num_colors == 2:
        #     dark_red = "#8B0000"
        #     return [base_color, dark_red]

        base_rgb = np.array(mcolors.to_rgb(base_color))
        base_hls = colorsys.rgb_to_hls(*base_rgb)

        lightness_values = np.linspace(
            base_hls[1], min(1.0, base_hls[1] + 0.4), num_colors)

        palette = [
            mcolors.to_hex(colorsys.hls_to_rgb(
                base_hls[0], lightness, base_hls[2]))
            for lightness in lightness_values
        ]

        return palette
    
    def _generate_param_palette(self, param_names, cmap_name="viridis"):
        """
        Return a list of colors from a colormap, one per parameter.

        Args:
            param_names (list): Names of parameters.
            cmap_name (str): Matplotlib colormap name, default 'viridis'.

        Returns:
            dict: Mapping {param_name: color}
        """
        cmap = plt.get_cmap(cmap_name)
        n = len(param_names)
        colors = [cmap(i) for i in np.linspace(0, 1, n)]
        return dict(zip(param_names, colors))

    def plot_surrogate_characteristics(self, surrogate: Type) -> None:
        """
        Plots the characteristics of the surrogate model, including the input and output dimensions, the number of training samples, and the number of parameters.

        Args:
            surrogate (Type): Surrogate model to visualize.
        """

        var_labels = {
            "c_0": "First PCE coefficient ($c_0$)",
            "c": "PCE coefficients ($c$)",
            "sigma": "Std of PCE Likelihood ($\sigma$)"
        }

        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.alpha"] = 0.6

        inference_data = az.from_cmdstanpy(surrogate.fit)
        inference_data = inference_data.rename(
            {"c_0": var_labels["c_0"], "c": var_labels["c"], "sigma": var_labels["sigma"]})
        fig = az.plot_trace(inference_data, var_names=[
                            var_labels["c_0"], var_labels["c"], var_labels["sigma"]])

        plt.subplots_adjust(top=0.85)
        plt.suptitle("Surrogate Posterior Distributions & Traces",
                     fontsize=14, y=0.98)
        plt.tight_layout()

        plt.savefig(os.path.join(self._config.output_path,
                    "figures/surrogate_characteristics.pdf"))

    def plot_apc_surrogate_characteristics(self, surrogate: Type) -> None:
        """
        Plots the characteristics of the apc surrogate model, including the input and output dimensions, the number of training samples, and the number of parameters.

        Args:
            surrogate (Type): Surrogate model to visualize.
        """

        var_labels = {
            "c_0": "First PCE coefficient ($c_0$)",
            "c": "PCE coefficients ($c$)",
            "sigma": "Std of PCE Likelihood ($\sigma$)"
        }

        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.alpha"] = 0.6
        for t_idx, time in enumerate(surrogate.times):
            for l_idx, location in enumerate(surrogate.locations):
                inference_data = az.from_cmdstanpy(
                    surrogate.fits[t_idx][l_idx])
                inference_data = inference_data.rename(
                    {"c_0": var_labels["c_0"], "c": var_labels["c"], "sigma": var_labels["sigma"]})
                fig = az.plot_trace(inference_data, var_names=[
                                    var_labels["c_0"], var_labels["c"], var_labels["sigma"]])

                plt.subplots_adjust(top=0.85)
                plt.suptitle(
                    "Surrogate Posterior Distributions & Traces", fontsize=14, y=0.98)
                plt.tight_layout()

                plt.savefig(os.path.join(self._config.output_path, "figures",
                                         f"surrogate_characteristics_time_{time}_location_{location}.pdf"))
                plt.close()

    def plot_output_draws(self, model: Type, surrogate: Type) -> None:
        """
        Plots the output draws of the Bayesian simulation model and the surrogate model.

        Args:
            model (Type): Simulation model.
            surrogate (Type): Surrogate model.
        """

        inputs = model.get_plot_inputs(
            self._config.out.n_plot_params, self._config.out.n_plot_inputs)

        if len(inputs) == 1:

            out_true = model.evaluate(*inputs)
            out_pce = surrogate.evaluate_full_posterior_predictive(*inputs)

            out_pce_q05 = np.quantile(out_pce, axis=0, q=0.05)
            out_pce_q95 = np.quantile(out_pce, axis=0, q=0.95)

            fig = plt.figure(figsize=(10, 7))
            plt.plot(inputs[0], out_true, "-", label="Simulation Model")
            plt.plot(inputs[0], out_pce.mean(axis=0), "-",
                     color="red", label="Surrogate Model (Mean)")
            plt.fill_between(inputs[0], out_pce_q05, out_pce_q95,
                             color="red", alpha=0.2)

            plt.xlabel("Input", fontsize=22)
            plt.ylabel("Output", fontsize=22)
            plt.title("Simulation & Surrogate Model Output",
                      fontsize=22, y=1.02)

            plt.legend(loc="best", fontsize=16)

        elif len(inputs) == 2:

            P, I = np.meshgrid(inputs[0], inputs[1], indexing="ij")

            input_pairs = np.column_stack([P.ravel(), I.ravel()])

            out_true = model.evaluate(*input_pairs.T)
            out_pce = surrogate.evaluate_full_posterior_predictive(
                *input_pairs.T)

            out_pce_mean = out_pce.mean(axis=0)
            out_pce_q05 = np.quantile(out_pce, axis=0, q=0.05)
            out_pce_q95 = np.quantile(out_pce, axis=0, q=0.95)

            out_true = out_true.reshape(
                self._config.out.n_plot_params, self._config.out.n_plot_inputs)
            out_pce = out_pce.reshape(
                out_pce.shape[0], self._config.out.n_plot_params, self._config.out.n_plot_inputs)
            out_pce_mean = out_pce_mean.reshape(
                self._config.out.n_plot_params, self._config.out.n_plot_inputs)
            out_pce_q05 = out_pce_q05.reshape(
                self._config.out.n_plot_params, self._config.out.n_plot_inputs)
            out_pce_q95 = out_pce_q95.reshape(
                self._config.out.n_plot_params, self._config.out.n_plot_inputs)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot_surface(I, P, out_true, cmap="Blues",
                            alpha=0.7, label="Simulation Model")
            ax.plot_surface(I, P, out_pce_mean, color="red",
                            alpha=0.5, label="Surrogate Model (Mean)")

            ax.plot_surface(I, P, out_pce_q05, color="red",
                            alpha=0.1, label="Surrogate Model (90% CI)")
            ax.plot_surface(I, P, out_pce_q95, color="red", alpha=0.1)

            ax.set_xlabel("Input", fontsize=22)
            ax.set_ylabel("Parameter", fontsize=22)
            ax.set_zlabel("Output", fontsize=22)
            ax.set_title("Simulation & Surrogate Model Output", fontsize=22)

            plt.legend(loc="best", fontsize=16)

        else:
            raise ValueError("Input shape not supported for plotting.")

        plt.savefig(os.path.join(
            self._config.output_path, "figures/outputs.pdf"))

    def plot_outputs(self, model: Type, surrogate: Type) -> None:
        """
        Plots the output of the simulation model and the surrogate model.

        Args:
            model (Type): Simulation model.
            surrogate (Type): Surrogate model.
        """

        inputs = model.get_plot_inputs(
            self._config.out.n_plot_params, self._config.out.n_plot_inputs)

        if len(inputs) == 1:

            out_true = model.evaluate(*inputs)
            plt.plot(inputs[0], out_true, "-", label="Simulation Model")
            plt.title("Simulation Model Output", fontsize=20, y=1.02)
            if surrogate is not None:
                out_pce = surrogate.evaluate(*inputs)
                plt.plot(inputs[0], out_pce, "-",
                         color="red", label="Surrogate Model")
                plt.title("Simulation & Surrogate Model Output",
                          fontsize=22, y=1.02)

            plt.xlabel("Input", fontsize=22)
            plt.ylabel("Output", fontsize=20)

            plt.legend(loc="best", fontsize=16)

        elif len(inputs) == 2:

            P, I = np.meshgrid(inputs[0], inputs[1], indexing="ij")

            input_pairs = np.column_stack([P.ravel(), I.ravel()])

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            out_true = model.evaluate(*input_pairs.T)
            out_true = out_true.reshape(
                self._config.out.n_plot_params, self._config.out.n_plot_inputs)
            ax.plot_surface(I, P, out_true, cmap="Blues",
                            alpha=0.7, label="Simulation Model")
            ax.set_title("Simulation Model Output")
            if surrogate is not None:
                out_pce = surrogate.evaluate(*input_pairs.T)
                out_pce = out_pce.reshape(
                    self._config.out.n_plot_params, self._config.out.n_plot_inputs)
                ax.plot_surface(I, P, out_pce, color="red",
                                alpha=0.5, label="Surrogate Model")
                ax.set_title(
                    "Simulation & Surrogate Model Output", fontsize=16)

            ax.set_xlabel("Input", fontsize=22)
            ax.set_ylabel("Parameter", fontsize=22)
            ax.set_zlabel("Output", fontsize=22)

            plt.legend(loc="best", fontsize=16)

        else:
            raise ValueError("Input shape not supported for plotting.")

        plt.savefig(os.path.join(
            self._config.output_path, "figures/outputs.pdf"))

    def plot_abi_loss(self, history: Dict[str, Any]) -> None:
        """
        Plots the loss history of the ABI model during training.

        Args:
            history (Dict[str, Any]): Dictionary containing the loss history of the ABI model.
        """

        fig = bf.diagnostics.plots.loss(history, train_key="loss")

        plt.savefig(os.path.join(self._config.output_path, "figures/loss.pdf"))

    def plot_posterior_parameter_sample(self, model: Type, out_params: Dict, parameter_draws: np.ndarray, base_file_name: str="parameter_posterior") -> None:
        """
        Plots the parameter posteriors of the posterior parameter draws.

        Args:
            model (Type): Model.
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
        """

        assert self._config.out.n_plot_param_samples <= self._config.out.n_datasets, "Cannot plot more posterior samples than datasets created."

        for sample in np.arange(self._config.out.n_plot_param_samples):
            if sample >= parameter_draws[model.get_param_names()[0]].shape[0]:
                return
            parameter_draws_sample = {
                p: parameter_draws[p][sample, :] for p in model.get_param_names()}

            fig = bf.diagnostics.plots.pairs_samples(
                parameter_draws_sample, variable_names=model.get_param_names())
            if out_params is not None:
                for i in range(len(out_params.keys())):
                    ax = fig.axes[i, i]
                    ax.axvline(out_params[list(out_params.keys())[
                            i]][0], color="red", linestyle="-", linewidth=1, label="True Value")
                    ax.legend(fontsize=16)

            plt.savefig(os.path.join(self._config.output_path,
                        f"figures/{base_file_name}_{sample}.pdf"))

    def plot_posterior_parameter_recovery(self, model: Type, out_datasets: np.ndarray, parameter_draws: np.ndarray) -> None:
        """
        Plots the parameter recovery of the posterior parameter draws.

        Args:
            model (Type): Model.
            out_datasets (np.ndarray): Output datasets from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
        """

        parameter_dict = {p: parameter_draws[p]
                          for p in model.get_param_names()}
        out_datasets_dict = {p: out_datasets[p]
                             for p in model.get_param_names()}

        fig = bf.diagnostics.plots.recovery(
            estimates=parameter_dict, targets=out_datasets_dict)
        plt.savefig(os.path.join(self._config.output_path,
                    "figures/parameter_recovery.pdf"))

    def plot_posterior_parameter_calibration(self, model: Type, out_datasets: np.ndarray, parameter_draws: np.ndarray) -> None:
        """
        Plots the parameter calibration ECDF of the posterior parameter draws.

        Args:
            model (Type): Model.
            out_datasets (np.ndarray): Output datasets from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
        """

        parameter_dict = {p: parameter_draws[p]
                          for p in model.get_param_names()}
        out_datasets_dict = {p: out_datasets[p]
                             for p in model.get_param_names()}

        fig = bf.diagnostics.plots.calibration_ecdf(
            estimates=parameter_dict, targets=out_datasets_dict, difference=True, rank_type="fractional")
        plt.savefig(os.path.join(self._config.output_path,
                    "figures/parameter_calibration.pdf"))

    def compare_posterior_parameter_sample(self, out_params: Dict, parameter_draws: Dict[str, Dict[str, np.ndarray]], n_plot_param_samples: int, height: int = 5, labels: Dict = None, hist: bool = False) -> None:
        """
        Compares the parameter posteriors of the posterior parameter draws of several methods.

        Args:
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (Dict[str, Dict[str, np.ndarray]]): Parameter draws from each method and for each parameter.
            n_plot_param_samples (int): Number of parameter samples to plot.
            height (int): Height of figure (default: 5).
            labels (Dict): Labels for the methods and parameters.
            hist (bool): Whether to plot histograms (default: False).
        """

        height = height
        color = "#132a70"

        sizes = {v.shape for method in parameter_draws.values()
                 for v in method.values()}
        assert len(
            sizes) == 1, "All methods should have the same number of parameters and parameter shapes."

        for param in next(iter(parameter_draws.values())).keys():
            param_shapes = {
                parameter_draws[method][param].shape for method in parameter_draws}
        assert len(
            param_shapes) == 1, f"Parameter '{param}' has different shapes across methods."

        param_names = list(next(iter(parameter_draws.values())).keys())
        method_names = list(parameter_draws.keys())

        palette = self._generate_shades(color, len(method_names))
        method_colors = {method: palette[i]
                         for i, method in enumerate(method_names)}

        sns.set_theme(style="whitegrid")

        for sample in np.arange(n_plot_param_samples):
            fig, axes = plt.subplots(1, len(param_names), figsize=(
                height * len(param_names), height))

            if len(param_names) == 1:
                axes = [axes]

            for idx, param in enumerate(param_names):
                for method in method_names:

                    if labels is not None:
                        label = labels[method][param]
                    else:
                        label = method

                    param_array = parameter_draws[method][param][sample, :]
                    color = method_colors[method]
                    sns.kdeplot(param_array.squeeze(),
                                ax=axes[idx], label=label, color=color)
                    if hist:
                        sns.histplot(param_array.squeeze(
                        ), ax=axes[idx], color=color, alpha=0.5, stat="density")

                true_value = out_params[param][0]
                axes[idx].axvline(true_value, color="red",
                                  linestyle="-", linewidth=1, label="True Value")

                # axes[idx].set_title(f"{param}", fontsize=20, y=1.02)
                axes[idx].legend(fontsize=16)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

            plt.tight_layout()

        plt.show()

    def compare_posterior_parameter_recovery(self, out_params: Dict, parameter_draws: Dict[str, Dict[str, np.ndarray]], height: int = 5, labels: Dict = None, parameter_key: str | None = None) -> None:
        """
        Compares the parameter recovery of the posterior parameter draws of several methods.

        Args:
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
            height (int): Height of figure (default: 5).
            labels (Dict): Labels for the methods and parameters.
        """

        height = height
        color = "#132a70"

        sizes = {v.shape for method in parameter_draws.values()
                 for v in method.values()}
        assert len(
            sizes) == 1, "All methods should have the same number of parameters and parameter shapes."

        for param in next(iter(parameter_draws.values())).keys():
            param_shapes = {
                parameter_draws[method][param].shape for method in parameter_draws}
        assert len(
            param_shapes) == 1, f"Parameter '{param}' has different shapes across methods."

        model_names = list(next(iter(parameter_draws.values())).keys())
        inf_method_names = list(parameter_draws.keys())

        palette = self._generate_shades(color, len(inf_method_names))
        inf_method_colors = {
            inf_method: palette[i] for i, inf_method in enumerate(inf_method_names)}

        sns.set_theme(style="whitegrid")

        # fig, axes = plt.subplots(len(model_names), 1, figsize=(height, height * len(model_names)))
        fig, axes = plt.subplots(1, len(model_names), figsize=(
            height * len(model_names), height))

        if len(model_names) == 1:
            axes = [axes]

        for idx, model in enumerate(model_names):

            param_array = out_params[model]

            for method in inf_method_names:
                parameter_draws_model = parameter_draws[method][model]
                color = inf_method_colors[method]

                point_estimate = np.median(parameter_draws_model, axis=1)

                u = scipy.stats.median_abs_deviation(
                    parameter_draws_model, axis=1)

                if labels is not None:
                    label = labels[method][model]
                else:
                    label = method

                axes[idx].errorbar(param_array[:, 0], point_estimate[:, 0],
                                   yerr=u[:, 0], fmt="o", alpha=0.5, color=color, label=label)

                make_quadratic(axes[idx], param_array[:, 0],
                               point_estimate[:, 0])

                corr = np.corrcoef(
                    param_array[:, 0], point_estimate[:, 0])[0, 1]

            axes[idx].set_xlabel("Ground Truth Parameter", fontsize=22)
            if idx == 0:
                axes[idx].set_ylabel(
                    "Inferred Parameter Posterior", fontsize=22)
            # axes[idx].set_title(f"{param}", fontsize=20, y=1.02)
            axes[idx].legend(fontsize=16)

            axes[idx].tick_params(axis="both", labelsize=16)

            plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,
                    f"figures/{self._config.model}_recovery_{parameter_key}.pdf"))
        plt.show()

    def compare_posterior_multi_parameter_recovery(self, out_params: Dict, parameter_draws: Dict[str, Dict[str, np.ndarray]], height: int = 5, labels: Dict = None) -> None:
        """
        Compares the parameter recovery of the posterior parameter draws of several methods.

        Args:
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
            height (int): Height of figure (default: 5).
            labels (Dict): Labels for the methods and parameters.
        """

        height = height
        color = "#132a70"

        param_names = list(
            next(iter(next(iter(parameter_draws.values())).values())).keys())
        model_names = list(next(iter(parameter_draws.values())).keys())
        inf_method_names = list(parameter_draws.keys())

        palette = self._generate_shades(color, len(inf_method_names))
        inf_method_colors = {
            inf_method: palette[i] for i, inf_method in enumerate(inf_method_names)}

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(len(model_names), len(param_names), figsize=(
            height * len(param_names), height * len(model_names)), squeeze=False)


        for model_idx, model in enumerate(model_names):
            for param_idx, param in enumerate(param_names):
                param_array = out_params[model][param]
                for method in inf_method_names:
                    parameter_draws_i = parameter_draws[method][model][param]
                    color = inf_method_colors[method]

                    point_estimate = np.median(parameter_draws_i, axis=1)

                    u = scipy.stats.median_abs_deviation(
                        parameter_draws_i, axis=1)

                    if labels is not None:
                        label = labels[method][model]
                    else:
                        label = method

                    axes[model_idx, param_idx].errorbar(
                        param_array[:, 0], point_estimate[:, 0], yerr=u[:, 0], fmt="o", alpha=0.5, color=color, label=label)

                    make_quadratic(axes[model_idx, param_idx],
                                   param_array[:, 0], point_estimate[:, 0])

                    corr = np.corrcoef(
                        param_array[:, 0], point_estimate[:, 0])[0, 1]

                if model_idx == len(model_names) - 1:
                    axes[model_idx, param_idx].set_xlabel(
                        f"Ground Truth Parameter", fontsize=22)
                if param_idx == 0:
                    axes[model_idx, param_idx].set_ylabel(
                        f"Inferred Parameter Posterior", fontsize=22)
                # axes[param_idx, model_idx].set_title(f"{param}", fontsize=20, y=1.02)
                axes[model_idx, param_idx].legend(fontsize=16)

                axes[model_idx, param_idx].tick_params(
                    axis="both", labelsize=16)

                plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,
                    f"figures/parameters_recovery.pdf"))
        plt.show()

    def compare_posterior_parameter_mcmc_abi(self, out_params: Dict, parameter_draws: Dict[str, Dict[str, np.ndarray]], height: int = 5, labels: Dict = None) -> None:
        """
        Compares the parameter recovery of the posterior parameter draws of several methods.

        Args:
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
            height (int): Height of figure (default: 5).
            labels (Dict): Labels for the methods and parameters.
        """

        height = height
        color = "#132a70"

        sizes = {v.shape for method in parameter_draws.values()
                 for v in method.values()}
        assert len(
            sizes) == 1, "All methods should have the same number of parameters and parameter shapes."

        for param in next(iter(parameter_draws.values())).keys():
            param_shapes = {
                parameter_draws[method][param].shape for method in parameter_draws}
        assert len(
            param_shapes) == 1, f"Parameter '{param}' has different shapes across methods."

        param_names = list(next(iter(parameter_draws.values())).keys())
        method_names = list(parameter_draws.keys())

        palette = self._generate_shades(color, len(method_names))
        method_colors = {method: palette[i]
                         for i, method in enumerate(method_names)}

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(1, len(param_names), figsize=(
            height * len(param_names), height))

        if len(param_names) == 1:
            axes = [axes]

        for idx, param in enumerate(param_names):

            param_array = out_params[param]

            # for method in method_names:
            key_abi = next(
                (method for method in parameter_draws if "ABI" in method), None)
            parameter_draws_param_abi = parameter_draws[key_abi][param]
            key_mcmc = next(
                (method for method in parameter_draws if "MCMC" in method), None)
            parameter_draws_param_mcmc = parameter_draws[key_mcmc][param]

            if labels is not None:
                label_abi = labels[key_abi][param]
                label_mcmc = labels[key_mcmc][param]
            else:
                label_abi = key_abi
                label_mcmc = key_mcmc

            point_estimate_abi = np.median(parameter_draws_param_abi, axis=1)
            point_estimate_mcmc = np.median(parameter_draws_param_mcmc, axis=1)
            color = method_colors[key_abi]

            u_abi = scipy.stats.median_abs_deviation(
                parameter_draws_param_abi, axis=1)
            u_mcmc = scipy.stats.median_abs_deviation(
                parameter_draws_param_mcmc, axis=1)
            axes[idx].errorbar(point_estimate_mcmc[:, 0], point_estimate_abi[:, 0],
                               xerr=u_mcmc[:, 0], yerr=u_abi[:, 0], fmt="o", alpha=0.5, color=color)
            axes[idx].scatter(point_estimate_mcmc[:, 0],
                              point_estimate_abi[:, 0], color=color, alpha=0.5, marker='o')

            make_quadratic(axes[idx], point_estimate_mcmc[:,
                           0], point_estimate_abi[:, 0])

            corr = np.corrcoef(
                point_estimate_mcmc[:, 0], point_estimate_abi[:, 0])[0, 1]

            axes[idx].set_xlabel(label_mcmc, fontsize=22)
            axes[idx].set_ylabel(label_abi, fontsize=22)
            # axes[idx].set_title(f"{param}", fontsize=22, y=1.02)
            axes[idx].tick_params(axis="both", labelsize=16)

            plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,
                    f"figures/{self._config.model}_mcmc_vs_abi.pdf"))
        plt.show()

    def compare_posterior_multi_parameter_mcmc_abi(self, out_params: Dict, parameter_draws: Dict[str, Dict[str, np.ndarray]], height: int = 5, labels: Dict = None) -> None:
        """
        Compares the parameter recovery of the posterior parameter draws of several methods.

        Args:
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
            height (int): Height of figure (default: 5).
            labels (Dict): Labels for the methods and parameters.
        """

        height = height
        color = "#132a70"

        param_names = list(
            next(iter(next(iter(parameter_draws.values())).values())).keys())
        model_names = list(next(iter(parameter_draws.values())).keys())
        inf_method_names = list(parameter_draws.keys())

        palette = self._generate_shades(color, len(inf_method_names))
        inf_method_colors = {
            inf_method: palette[i] for i, inf_method in enumerate(inf_method_names)}

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(len(model_names), len(param_names), figsize=(
            height * len(param_names), height * len(model_names)), squeeze=False)

        for model_idx, model in enumerate(model_names):
            for param_idx, param in enumerate(param_names):

                key_abi = next(
                    (method for method in parameter_draws if "ABI" in method), None)
                parameter_draws_param_abi = parameter_draws[key_abi][model][param]
                key_mcmc = next(
                    (method for method in parameter_draws if "MCMC" in method), None)
                parameter_draws_param_mcmc = parameter_draws[key_mcmc][model][param]

                if labels is not None:
                    label_abi = labels[key_abi][model]
                    label_mcmc = labels[key_mcmc][model]
                else:
                    label_abi = key_abi
                    label_mcmc = key_mcmc

                point_estimate_abi = np.median(
                    parameter_draws_param_abi, axis=1)
                point_estimate_mcmc = np.median(
                    parameter_draws_param_mcmc, axis=1)
                color = inf_method_colors[key_abi]

                u_abi = scipy.stats.median_abs_deviation(
                    parameter_draws_param_abi, axis=1)
                u_mcmc = scipy.stats.median_abs_deviation(
                    parameter_draws_param_mcmc, axis=1)
                axes[model_idx, param_idx].errorbar(
                    point_estimate_mcmc[:, 0], point_estimate_abi[:, 0], xerr=u_mcmc[:, 0], yerr=u_abi[:, 0], fmt="o", alpha=0.5, color=color)
                axes[model_idx, param_idx].scatter(
                    point_estimate_mcmc[:, 0], point_estimate_abi[:, 0], color=color, alpha=0.5, marker='o')

                make_quadratic(
                    axes[model_idx, param_idx], point_estimate_mcmc[:, 0], point_estimate_abi[:, 0])

                corr = np.corrcoef(
                    point_estimate_mcmc[:, 0], point_estimate_abi[:, 0])[0, 1]

                axes[model_idx, param_idx].set_xlabel(label_mcmc, fontsize=22)
                axes[model_idx, param_idx].set_ylabel(label_abi, fontsize=22)
                # axes[model_idx, param_idx].set_title(f"{param}", fontsize=22, y=1.02)
                axes[model_idx, param_idx].tick_params(
                    axis="both", labelsize=16)

                plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,
                    f"figures/parameters_mcmc_vs_abi.pdf"))
        plt.show()

    def compare_posterior_ci_mcmc_abi(self, out_params: Dict, parameter_draws: Dict[str, Dict[str, np.ndarray]], height: int = 5) -> None:
        """
        Compares the ci recovery of the posterior parameter draws of several methods.

        Args:
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
            height (int): Height of figure (default: 5).
        """

        height = height
        color = "#132a70"

        sizes = {v.shape for method in parameter_draws.values()
                 for v in method.values()}
        assert len(
            sizes) == 1, "All methods should have the same number of parameters and parameter shapes."

        for param in next(iter(parameter_draws.values())).keys():
            param_shapes = {
                parameter_draws[method][param].shape for method in parameter_draws}
        assert len(
            param_shapes) == 1, f"Parameter '{param}' has different shapes across methods."

        param_names = list(next(iter(parameter_draws.values())).keys())
        method_names = list(parameter_draws.keys())

        palette = self._generate_shades(color, len(method_names))
        method_colors = {method: palette[i]
                         for i, method in enumerate(method_names)}

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(len(param_names), 1, figsize=(
            height, height * len(param_names)))

        if len(param_names) == 1:
            axes = [axes]

        for idx, param in enumerate(param_names):
            key_abi = next(
                (method for method in parameter_draws if "ABI" in method), None)
            parameter_draws_param_abi = parameter_draws[key_abi][param]
            key_mcmc = next(
                (method for method in parameter_draws if "MCMC" in method), None)
            parameter_draws_param_mcmc = parameter_draws[key_mcmc][param]

            color = method_colors[key_mcmc]

            u_abi = scipy.stats.median_abs_deviation(
                parameter_draws_param_abi, axis=1)
            u_mcmc = scipy.stats.median_abs_deviation(
                parameter_draws_param_mcmc, axis=1)
            axes[idx].plot(u_mcmc[:, 0], u_abi[:, 0], "o")

            make_quadratic(axes[idx], u_mcmc[:, 0], u_abi[:, 0])

            axes[idx].set_xlabel("CI " + key_mcmc, fontsize=22)
            axes[idx].set_ylabel("CI " + key_abi, fontsize=22)
            # axes[idx].set_title(f"{param}", fontsize=22, y=1.02)
            axes[idx].tick_params(axis="both", labelsize=16)

            plt.tight_layout()

        plt.show()

    def compare_posterior_parameter_calibration(self, out_params: Dict, parameter_draws: Dict[str, Dict[str, np.ndarray]], height: int = 5, labels: Dict = None, parameter_key: str | None = None) -> None:
        """
        Compares the parameter recovery of the posterior parameter draws of several methods.

        Args:
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
            height (int): Height of figure (default: 5).
            labels (Dict): Labels for the methods and parameters.
        """

        height = height
        color = "#132a70"

        sizes = {v.shape for method in parameter_draws.values()
                 for v in method.values()}
        assert len(
            sizes) == 1, "All methods should have the same number of parameters and parameter shapes."

        for param in next(iter(parameter_draws.values())).keys():
            param_shapes = {
                parameter_draws[method][param].shape for method in parameter_draws}
        assert len(
            param_shapes) == 1, f"Parameter '{param}' has different shapes across methods."

        param_names = list(next(iter(parameter_draws.values())).keys())
        method_names = list(parameter_draws.keys())

        palette = self._generate_shades(color, len(method_names))
        method_colors = {method: palette[i]
                         for i, method in enumerate(method_names)}

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(
            1, len(param_names), figsize=(height * len(param_names), 6))

        if len(param_names) == 1:
            axes = [axes]

        for idx, param in enumerate(param_names):

            param_array = out_params[param]

            for method in method_names:
                parameter_draws_param_method = parameter_draws[method][param]
                color = method_colors[method]

                ranks = fractional_ranks(
                    parameter_draws_param_method, param_array)

                for j in range(ranks.shape[-1]):
                    xx = np.repeat(np.sort(ranks[:, j]), 2)
                    xx = np.pad(xx, (1, 1), constant_values=(0, 1))
                    yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
                    yy = np.repeat(yy, 2)
                    yy -= xx

                    if labels is not None:
                        label = labels[method][param]
                    else:
                        label = method

                    axes[idx].plot(xx, yy, color=color,
                                   alpha=0.95, label=f"{label}")

                alpha, z, L, H = simultaneous_ecdf_bands(param_array.shape[0])

                L -= z
                H -= z

            # , label=rf"{int((1 - alpha) * 100)}$\%$ Confidence Bands")
            axes[idx].fill_between(z, L, H, color="grey", alpha=0.2)

            axes[idx].set_xlabel("Fractional Rank Statistic", fontsize=22)
            if idx == 0:
                axes[idx].set_ylabel("ECDF Difference", fontsize=22)
            # axes[idx].set_title(f"{param}", fontsize=20, y=1.02)
            axes[idx].tick_params(axis="both", labelsize=16)
            axes[idx].legend(loc="lower left", fontsize=16)

            plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,
                    f"figures/{self._config.model}_ecdf_{parameter_key}.pdf"))
        plt.show()

    def compare_posterior_multi_parameter_calibration(self, out_params: Dict, parameter_draws: Dict[str, Dict[str, np.ndarray]], height: int = 5, labels: Dict = None) -> None:
        """
        Compares the parameter recovery of the posterior parameter draws of several methods.

        Args:
            out_params (Dict): Output parameters from the simulator.
            parameter_draws (np.ndarray): Parameter draws from the model.
            height (int): Height of figure (default: 5).
            labels (Dict): Labels for the methods and parameters.
        """

        height = height
        color = "#132a70"

        param_names = list(
            next(iter(next(iter(parameter_draws.values())).values())).keys())
        model_names = list(next(iter(parameter_draws.values())).keys())
        method_names = list(parameter_draws.keys())

        palette = self._generate_shades(color, len(method_names))
        method_colors = {method: palette[i]
                         for i, method in enumerate(method_names)}

        sns.set_theme(style="whitegrid")

        fig, axes = plt.subplots(len(model_names), len(param_names), figsize=(
            height * len(param_names), height * len(model_names)), squeeze=False)


        for model_idx, model in enumerate(model_names):
            for param_idx, param in enumerate(param_names):
                param_array = out_params[model][param]

                for method in method_names:
                    parameter_draws_i = parameter_draws[method][model][param]
                    color = method_colors[method]

                    ranks = fractional_ranks(parameter_draws_i, param_array)

                    for j in range(ranks.shape[-1]):
                        xx = np.repeat(np.sort(ranks[:, j]), 2)
                        xx = np.pad(xx, (1, 1), constant_values=(0, 1))
                        yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
                        yy = np.repeat(yy, 2)
                        yy -= xx

                        if labels is not None:
                            label = labels[method][model]
                        else:
                            label = method

                        axes[model_idx, param_idx].plot(
                            xx, yy, color=color, alpha=0.95, label=f"{label}")

                    alpha, z, L, H = simultaneous_ecdf_bands(
                        param_array.shape[0])

                    L -= z
                    H -= z

                # , label=rf"{int((1 - alpha) * 100)}$\%$ Confidence Bands")
                axes[model_idx, param_idx].fill_between(
                    z, L, H, color="grey", alpha=0.2)
                if model_idx == len(model_names) - 1:
                    axes[model_idx, param_idx].set_xlabel(
                        "Fractional Rank Statistic", fontsize=22)
                if param_idx == 0:
                    axes[model_idx, param_idx].set_ylabel(
                        "ECDF Difference", fontsize=22)
                # axes[model_idx, param_idx].set_title(f"{param}", fontsize=20, y=1.02)
                axes[model_idx, param_idx].tick_params(
                    axis="both", labelsize=16)
                axes[model_idx, param_idx].legend(
                    loc="lower left", fontsize=16)

                plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,
                    f"figures/parameters_ecdf.pdf"))
        plt.show()

    def compare_runtimes_paths(self, results_paths: List[str]) -> None:
        """
        Compares the runtimes of different methods by given results paths.

        Args:
            results_paths (List[str]): Path to results folders.
        """

        color = "#132a70"

        times_abi = {}
        times_mcmc = {}

        for path in results_paths:

            assert os.path.exists(path), f"Path '{path}' does not exist."

            config = u.read_config(glob.glob(os.path.join(path, "*.json"))[0])

            with open(os.path.join(path, "diagnosis/time.json"), "r") as file:
                time_file = json.load(file)

            n_datasets = int(config.out.n_datasets)

            if config.inference == "abi":
                times_abi[n_datasets] = float(
                    time_file["abi_training_time"]) + float(time_file["abi_sampling_time"])
            elif config.inference == "mcmc":
                times_mcmc[n_datasets] = float(time_file["mcmc_sampling_time"])
            else:
                raise ValueError("Inference method not supported.")

        if not times_abi and not times_mcmc:
            raise ValueError("Times not readable for given paths.")

        sns.set_theme(style="whitegrid")

        fig = plt.figure(figsize=(8, 5))

        palette = self._generate_shades(color, 2)
        if times_abi:
            plt.plot(times_abi.keys(), times_abi.values(), label="ABI",
                     marker="x", markersize=10, color=palette[0])
        if times_mcmc:
            plt.plot(times_mcmc.keys(), times_mcmc.values(),
                     label="MCMC", marker="x", markersize=10, color=palette[1])

        plt.xlabel("Number of Inference Runs", fontsize=22)
        plt.ylabel("Runtime [s]", fontsize=22)
        plt.title("Runtime Comparison", fontsize=22, y=1.02)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc="best", fontsize=16)

        plt.tight_layout()
        plt.show()

    def compare_runtimes(self, times_abi: Dict, times_mcmc: Dict) -> None:
        """
        Compares the runtimes of different methods by given runtimes.

        Args:
            times_abi (Dict): ABI runtimes.
            times_mcmc (Dict): MCMC runtimes.
        """

        color = "#132a70"

        sns.set_theme(style="whitegrid")

        fig = plt.figure(figsize=(8, 4))

        palette = self._generate_shades(color, 2)
        if times_abi:
            keys = sorted(times_abi.keys())
            values = [times_abi[k] for k in keys]
            plt.plot(keys, values, label="UA-SABI", marker="x",
                     markersize=10, color=palette[0])
        if times_mcmc:
            keys = sorted(times_mcmc.keys())
            values = [times_mcmc[k] for k in keys]
            plt.plot(keys, values, label="E-Post", marker="x",
                     markersize=10, color=palette[1])

        # Compute breakeven point
        keys = sorted(set(times_abi) & set(times_mcmc))
        y_abi = [times_abi[k] for k in keys]
        y_mcmc = [times_mcmc[k] for k in keys]

        # Find first index where lines cross
        breakeven_x = None
        for i in range(1, len(keys)):
            diff_prev = y_abi[i-1] - y_mcmc[i-1]
            diff_curr = y_abi[i] - y_mcmc[i]
            if diff_prev * diff_curr < 0:  # sign change: lines crossed
                # Optional: linear interpolation for more accuracy
                x1, x2 = keys[i-1], keys[i]
                y1_diff, y2_diff = diff_prev, diff_curr
                breakeven_x = x1 - y1_diff * (x2 - x1) / (y2_diff - y1_diff)
                break

        # Plot vertical line and label (if found)
        if breakeven_x is not None:
            plt.axvline(x=breakeven_x, color="orange",
                        linestyle="--", linewidth=2)

        plt.xlabel("Number of Inference Runs", fontsize=22)
        plt.ylabel("Runtime [s]", fontsize=22)
        # plt.title("Runtime Comparison", fontsize=16, y=1.02)
        plt.xticks(ticks=list(times_mcmc.keys()), fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc="best", fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,
                    f"figures/{self._config.model}_runtimes.pdf"))
        plt.show()

    def plot_sobols(self, sobol_maps: Dict, time: int = -1, ci: float = 0.9) -> None:
        """
        Plot (distribution of) Sobol indices for each parameter over all locations at specified time point.

        Args:
            sobol_maps (Dict): Dictionary containing Sobol indices.
            time (int): Time point to plot Sobol indices for. Default is -1 (last time point).
            ci (float): Confidence interval for the Sobol indices. Default is 0.9.
        """
        param_names = list(sobol_maps.keys())

        palette = self._generate_param_palette(param_names)

        sns.set_theme(style="whitegrid")
        plt.rcParams.update(self.rcParams)
        fig = plt.figure()

        for param in param_names:

            n_locs = sobol_maps[param].shape[1]

            sobol_median = np.zeros((n_locs,))
            sobol_upper = np.zeros((n_locs,)) 
            sobol_lower = np.zeros((n_locs,))   

            for l in range(n_locs):
                samples = sobol_maps[param][time, l, :]
                sobol_median[l] = np.median(samples)
                sobol_upper[l] = np.quantile(samples, 0.5 + ci/2.0)
                sobol_lower[l] = np.quantile(samples, 0.5 - ci/2.0)
                                                                    

            plt.plot(np.arange(1, n_locs+1), sobol_median, label=param, color=palette[param], linewidth=2)
            plt.fill_between(np.arange(1, n_locs+1), sobol_lower, sobol_upper, color=palette[param], alpha=0.2)
            
        plt.xlabel("Location Index")
        plt.ylabel("Total Sobol Index")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,f"figures/{self._config.model}_sobol_indices.pdf"))
        plt.show()

    def plot_comparison_single_posterior_kde(self, draws_dict: dict, param_to_plot: str = r"$k_{ub}$",
                                             methods: List[str] = ["mcmc_point", "sabi", "mcmc_e_post", "uasabi", "abi_low_budget"]) -> None:
        """
        Plots KDEs for param_4 with consistent color and line style encoding.

        Color encodes group (MCMC, ABI, Low-budget).
        Line style encodes whether the method uses point estimate or posterior.

        Args:
            draws_dict (dict): Dictionary with method names as keys
                            and param draw dicts as values.
        """
        color_palette = self._generate_shades("#132a70", 2)
        method_styles = {
            "mcmc_point":       {"label": "Point",       "color": color_palette[1],   "linestyle": "--"},
            "sabi":             {"label": "SABI",             "color": color_palette[0],   "linestyle": "--"},
            "mcmc_e_post":      {"label": "E-Post",      "color": color_palette[1], "linestyle": "-"},
            "uasabi":           {"label": "UA-SABI",          "color": color_palette[0], "linestyle": "-"},
            "abi_low_budget":   {"label": "Low-budget ABI",   "color": "purple", "linestyle": "-."}
        }

        sns.set_theme(style="whitegrid")
        plt.rcParams.update(self.rcParams)
        plt.figure()
        plt.ticklabel_format(style="sci", axis="both", scilimits=(0,0), useMathText=True)

        for method in methods:
            if method not in draws_dict:
                continue
            if param_to_plot not in draws_dict[method]:
                continue

            param_vals = draws_dict[method][param_to_plot].squeeze()
            style = method_styles[method]

            if np.var(param_vals) > 1e-18:
                sns.kdeplot(
                    param_vals,
                    fill=True,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2
                )
            else:
                val = np.mean(param_vals)  # or param_vals[0]
                plt.axvline(
                    x=val,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2,
                    label=style["label"]
                )

        plt.xlabel(param_to_plot)
        plt.ylabel("Posterior KDE")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,f"figures/{self._config.model}_posterior_kde_{param_to_plot}_methods_{len(methods)}.pdf"))
        plt.show()

    def plot_comparison_multi_posterior_kde(
            self,
            draws_dict: dict,
            params_to_plot: List[str],
            methods: List[str] = ["mcmc_point", "sabi", "mcmc_e_post", "uasabi", "abi_low_budget"],
            ncols: int = 1,
            figsize_per_panel=(6, 4),
            sharex: bool = False,
            sharey: bool = False,
            save_name: str = None
            ) -> None:
        """
        Plots KDEs for multiple with consistent color and line style encoding.

        Color encodes group (MCMC, ABI, Low-budget).
        Line style encodes whether the method uses point estimate or posterior.

        Args:
            draws_dict (dict): Dictionary with method names as keys
                            and param draw dicts as values.
        """
        color_palette = self._generate_shades("#132a70", 2)
        method_styles = {
            "mcmc_point":       {"label": "Point",       "color": color_palette[1],   "linestyle": "--"},
            "sabi":             {"label": "SABI",             "color": color_palette[0],   "linestyle": "--"},
            "mcmc_e_post":      {"label": "E-Post",      "color": color_palette[1], "linestyle": "-"},
            "uasabi":           {"label": "UA-SABI",          "color": color_palette[0], "linestyle": "-"},
            "abi_low_budget":   {"label": "Low-budget ABI",   "color": "purple", "linestyle": "-."}
        }

        n_params = len(params_to_plot)
        nrows = math.ceil(n_params / ncols)
        fig_w = self.figsize[0] * ncols
        fig_h = self.figsize[1] * nrows
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(self.rcParams)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=sharex, sharey=sharey)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()
        for i, param in enumerate(params_to_plot):
            ax = axes[i]
            for method in methods:
                if method not in draws_dict:
                    continue
                if param not in draws_dict[method]:
                    continue

                param_vals = draws_dict[method][param].squeeze()
                style = method_styles[method]

                if (method == "abi_low_budget" and np.std(param_vals) > 1e-6) or (method != "abi_low_budget" and np.std(param_vals) > 1e-9):
                    sns.kdeplot(
                        param_vals,
                        ax=ax,
                        fill=True,
                        label=style["label"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2
                    )
                else:
                    val = np.mean(param_vals)  # or param_vals[0]
                    ax.axvline(
                        x=val,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2,
                        label=style["label"]
                    )
            ax.set_xlabel(param)
            ax.set_ylabel("Posterior KDE")
            ax.ticklabel_format(style="sci", axis="both", scilimits=(0,0), useMathText=True)
        plt.legend()
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(os.path.join(self._config.output_path, "figures", f"{save_name}.pdf"))
        plt.show()

    def plot_surr_prior_predictive(
            self,
            out_observations: np.ndarray,
            measurement: np.ndarray = None,
            locations: list | np.ndarray = None,
            measurement_sigma: np.ndarray = None,
            ci_level: float = 0.90,
            out_simulation: np.ndarray = None,
            median_kwargs: dict = None,
            ci_kwargs: dict = None,
            meas_kwargs: dict = None,
            err_kwargs: dict = None,
            sim_kwargs: dict = None,
            kub_upper: float = 0.5e-4,
            figsize: Tuple = (8, 5),
            ylim: Tuple = None,
            save_name: str = None
        ):
        """
        Plot the prior predictive *summary* across locations:
        - median curve
        - shaded credible interval (default: 90% CI)

        Args:
            out_observations (np.ndarray): Shape (n_samples, n_locations).
            measurement (np.ndarray, optional): Shape (n_locations,). Plotted as a line.
            locations (list | np.ndarray, optional): X-axis labels/positions (length n_locations).
            measurement_sigma (np.ndarray, optional): Per-location σ for error bars (length n_locations).
            ci_level (float): Width of the credible interval in (0,1]. Default 0.90 (i.e., 5–95%).
            simulation (np.ndarray, optional): Shape (n_locations,). Plotted as a line.
            median_kwargs (dict): Matplotlib kwargs for the median line.
            ci_kwargs (dict): Matplotlib kwargs for the CI band (passed to fill_between).
            meas_kwargs (dict): Matplotlib kwargs for the measurement line.
            err_kwargs (dict): Matplotlib kwargs for the measurement error bars.
        """
        if out_observations.ndim != 2:
            raise ValueError(f"`out_observations` must be 2D (n_samples, n_locations); got shape {out_observations.shape}.")

        n_samples, n_locations = out_observations.shape
        x = np.arange(1, n_locations + 1) if locations is None else np.asarray(locations)
        if x.shape[0] != n_locations:
            raise ValueError("`locations` length must match n_locations.")

        # summary stats
        lower_q = (1 - ci_level) / 2 * 100.0
        upper_q = (1 + ci_level) / 2 * 100.0
        median = np.median(out_observations, axis=0)
        lower = np.percentile(out_observations, lower_q, axis=0)
        upper = np.percentile(out_observations, upper_q, axis=0)

        # Defaults for plotting
        if median_kwargs is None:
            median_kwargs = dict(color=self.surrogate_color, linewidth=2.5, label=f"Surrogate") 
        if ci_kwargs is None:
            ci_kwargs = dict(alpha=0.25, color=self.surrogate_color)
        if meas_kwargs is None:
            meas_kwargs = dict(color=self.measurement_color, linewidth=2, alpha=0.7)
        if err_kwargs is None:
            err_kwargs = dict(fmt='x', color=self.measurement_color, elinewidth=1.5, capsize=4, label='Measurement')
        if sim_kwargs is None:
            sim_kwargs = dict(color=self.simulation_color, linewidth=1, alpha=0.7, linestyle='--', marker="o", markersize=3, label="Simulation")

        sns.set_style("whitegrid")

        plt.rcParams.update(self.rcParams)
        plt.figure(figsize=figsize)

        # CI band
        plt.fill_between(x, lower, upper, **ci_kwargs)

        # Median line
        plt.plot(x, median, **median_kwargs)

        # Simulation runs overlay
        if out_simulation is not None:
            if out_simulation.ndim != 2 or out_simulation.shape[1] != n_locations:
                raise ValueError(f"`out_simulation` must be shape (n_runs, {n_locations}), got {out_simulation.shape}.")
            for i, sim in enumerate(out_simulation):
                if i == 0:
                    plt.plot(x, sim, **sim_kwargs)  # keep label for first line
                else:
                    plt.plot(x, sim, **{k: v for k, v in sim_kwargs.items() if k != "label"})

        # Optional measurement overlay
        if measurement is not None:
            if measurement.shape[0] != n_locations:
                raise ValueError("`measurement` length must match n_locations.")
            plt.plot(x, measurement, **meas_kwargs)

            if measurement_sigma is not None:
                if measurement_sigma.shape[0] != n_locations:
                    raise ValueError("`measurement_sigma` length must match n_locations.")
                plt.errorbar(x, measurement, yerr=measurement_sigma, **err_kwargs)

        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel("Location Index" if locations is None else "Location")
        plt.ylabel("Calcite")
        plt.legend()
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(os.path.join(self._config.output_path,f"figures/{save_name}.pdf"))
        plt.show()

    def plot_surrogate_output_vs_kub(
            self,
            surrogate,
            out_params_template: dict,
            locations: list,
            location_index: int,
            param_4_values: np.ndarray,
            train_bounds: tuple,
            time: int = 1,
            measurement: np.ndarray = None,
            measurement_sigma: np.ndarray = None,
            param_names: np.ndarray = None
        ):
        """
        Plots surrogate output (median and 90% CI) at a fixed location
        as a function of param_4 only, highlighting extrapolation regions.

        Args:
            surrogate: Surrogate model with evaluate_full_posterior_predictive method.
            out_params_template (dict): Template dict with full parameter arrays.
            locations (list): List of locations.
            location_index (int): Index of the location to evaluate.
            param_4_values (np.ndarray): 1D array of values to scan for param_4.
            train_bounds (tuple): (min, max) tuple for param_4 training range.
            time (int): Time to evaluate the surrogate at.
        """
        def evaluate_surrogate_over_locations(
            surrogate,
            out_params: Dict[str, np.ndarray],
            locations: List[int],
            time: int = 1,
            num_post_samples: int = 1000
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            """
            Evaluates the surrogate model across all locations and parameter samples.

            Args:
                surrogate: Surrogate model with method `evaluate_full_posterior_predictive(...)`
                out_params (dict): Dict with keys 'param_1' to 'param_4', each of shape (N, 1)
                locations (list): List of location indices to evaluate
                time (int): Time point to use in surrogate (default: 1)
                num_post_samples (int): Number of posterior predictive samples per param sample

            Returns:
                Tuple of:
                    y_surs (np.ndarray): shape (num_locations, num_post_samples, num_param_samples)
                    y_surs_lower (np.ndarray): shape (num_locations, num_param_samples)
                    y_surs_median (np.ndarray): shape (num_locations, num_param_samples)
                    y_surs_upper (np.ndarray): shape (num_locations, num_param_samples)
            """
            num_param_samples = out_params[param_names[0]].shape[0]
            num_locations = len(locations)

            # Allocate arrays
            y_surs = np.zeros((num_locations, num_post_samples, num_param_samples))
            y_surs_lower = np.zeros((num_locations, num_param_samples))
            y_surs_median = np.zeros((num_locations, num_param_samples))
            y_surs_upper = np.zeros((num_locations, num_param_samples))

            # Flatten parameters to 1D arrays
            x1 = out_params[param_names[0]][:, 0]
            x2 = out_params[param_names[1]][:, 0]
            x3 = out_params[param_names[2]][:, 0]
            x4 = out_params[param_names[3]][:, 0]

            # Evaluate surrogate at each location
            for li, loc in enumerate(locations):
                y_sur = surrogate.evaluate_full_posterior_predictive(
                    x1, x2, x3, x4, location=loc, time=time
                )  # shape (num_post_samples, num_param_samples)

                y_surs[li, :, :] = y_sur
                y_surs_lower[li, :] = np.quantile(y_sur, 0.05, axis=0)
                y_surs_median[li, :] = np.quantile(y_sur, 0.50, axis=0)
                y_surs_upper[li, :] = np.quantile(y_sur, 0.95, axis=0)

            return y_surs, y_surs_lower, y_surs_median, y_surs_upper
        # Fix param_1, param_2, param_3
        p1_fixed = float(np.median(out_params_template[param_names[0]]))
        p2_fixed = float(np.median(out_params_template[param_names[1]]))
        p3_fixed = float(np.median(out_params_template[param_names[2]]))

        y_median = []
        y_lower = []
        y_upper = []

        for p4_val in param_4_values:
            out_params = {
                param_names[0]: np.array([[p1_fixed]]),
                param_names[1]: np.array([[p2_fixed]]),
                param_names[2]: np.array([[p3_fixed]]),
                param_names[3]: np.array([[p4_val]])
            }

            y_surs, y_surs_lower, y_surs_median, y_surs_upper = evaluate_surrogate_over_locations(
                surrogate=surrogate,
                out_params=out_params,
                locations=[locations[location_index]],
                time=time
            )

            y_median.append(y_surs_median[0, 0])
            y_lower.append(y_surs_lower[0, 0])
            y_upper.append(y_surs_upper[0, 0])

        y_median = np.array(y_median)
        y_lower = np.array(y_lower)
        y_upper = np.array(y_upper)

        # Plot
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(self.rcParams)
        plt.figure()
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
        plt.plot(param_4_values, y_median, label='Surrogate', color=self.surrogate_color)
        plt.fill_between(param_4_values, y_lower, y_upper, alpha=0.2, color=self.surrogate_color)
        plt.axhline(y=measurement[location_index], color=self.measurement_color, linestyle='--', label='Measurement')
        plt.axhline(y=measurement[location_index] + measurement_sigma[location_index], color=self.measurement_color, linestyle=':')
        plt.axhline(y=measurement[location_index] - measurement_sigma[location_index], color=self.measurement_color, linestyle=':')

        # Mark training region
        plt.axvspan(train_bounds[0], train_bounds[1], color='grey', alpha=0.15)

        plt.xlabel(param_names[3])
        plt.ylabel("Calcite")
        # plt.title(f"Surrogate output at location {locations[location_index]}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self._config.output_path,f"figures/{self._config.model}_location_surrogate_output.pdf"))
        plt.show()

