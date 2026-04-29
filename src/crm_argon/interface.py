#  Project: Collisional Radiative Model for Argon
#  Author: Peter Preisler
#  Contact: via contact form - https://peterplr.github.io/
#
#  Credits:
#  - Methodology based on: Vlcek (1989)
#  - Data sourced from: Kramida et al. (2024), Kimura et al. (1985), and Katsonis (1976)
#
#  Copyright (C) 2026 Peter Preisler
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
import numpy as np
import itertools
from scipy.optimize import minimize, Bounds
from itertools import combinations

# TOML parsing (Python 3.11+ natively, fallback for older)
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from .database import Database
from .crossSections import CrossSections
from .kinetics import Kinetics
from .solver import SteadyStateSolver
from .physicsEngine import PhysicsEngine
from .eedf import EEDF
from .diagnostics import FluxAnalyzer
from .constants import k_B
from .plotter import Plotter
from .exporter import Exporter


class Interface:
    """
    High-level API for the Collisional-Radiative (CR) Model of Argon.
    """

    # =========================================================================
    # CORE COMPONENTS INITIALIZATION
    # =========================================================================

    def __init__(self, model_config_file, run_config_file):
        """
        Initializes the CR model environment using TOML configuration files.

        Parameters:
            model_config_file (str): Path to the TOML model setup file.
            run_config_file (str): Path to the TOML run parameters file.
        """
        print("Loading Configuration Files...")

        with open(model_config_file, 'rb') as f:
            self.model_config = tomllib.load(f)

        with open(run_config_file, 'rb') as f:
            self.run_config = tomllib.load(f)

        self.eedf_mode = self.run_config.get("execution", {}).get("eedf_mode", "maxwellian").lower()
        self.output_dir = self.run_config.get("execution", {}).get("output_dir", "./output")

        # Determine paths
        custom_data_path = self.model_config.get("model", {}).get("custom_data_path")

        print("Initializing CR Model components...")
        self.db = Database(custom_data_path=custom_data_path)
        self.xs = CrossSections()
        self.calc = Kinetics()
        self.solver = SteadyStateSolver()

        # Inject the PARSED DICT into the Physics Engine
        self.physics = PhysicsEngine(
            config_dict=self.model_config,
            database=self.db,
            cross_sections=self.xs,
            rate_calculator=self.calc,
            steady_state_solver=self.solver
        )
        print("CR Model initialized successfully.\n")

    # =========================================================================
    # MAIN EXECUTION ROUTER
    # =========================================================================

    def execute(self):
        """
        Reads the run configuration mode and routes to either Simulation or Optimization.
        """
        mode = self.run_config.get("mode", "simulation").lower()

        if mode == "simulation":
            self._execute_simulation_sweep()
        elif mode == "optimization":
            self._execute_optimization()
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'simulation' or 'optimization'.")

    def _execute_simulation_sweep(self):
        """Handles parameter sweeps and executes the simulation for each condition."""
        print("=== Starting Simulation Mode ===")

        plasma_cfg = self.run_config["plasma"]
        dens_cfg = plasma_cfg["density"]

        # Extract parameters, wrapping scalars in lists to unify logic
        params = {
            'Te_eV': plasma_cfg['Te_eV'] if isinstance(plasma_cfg['Te_eV'], list) else [plasma_cfg['Te_eV']],
            'Tg_K': plasma_cfg['Tg_K'] if isinstance(plasma_cfg['Tg_K'], list) else [plasma_cfg['Tg_K']],
            'pg_Pa': plasma_cfg['pg_Pa'] if isinstance(plasma_cfg['pg_Pa'], list) else [plasma_cfg['pg_Pa']],
            'R_cm': plasma_cfg['R_cm'] if isinstance(plasma_cfg['R_cm'], list) else [plasma_cfg['R_cm']],
            'n_e_cm3': dens_cfg['n_e_cm3'] if isinstance(dens_cfg['n_e_cm3'], list) else [dens_cfg['n_e_cm3']],
            'n_ion_cm3': dens_cfg['n_ion_cm3'] if isinstance(dens_cfg['n_ion_cm3'], list) else [dens_cfg['n_ion_cm3']]
        }

        # Create a Cartesian product of all parameters (handles sweeps gracefully)
        keys, values = zip(*params.items())
        run_matrix = [dict(zip(keys, v)) for v in itertools.product(*values)]

        analyze_levels = self.model_config.get("levels", {}).get("analyze")

        # Store combined results for plotter
        all_results = []

        for conditions in run_matrix:
            # Calculate n_1 dynamically for this specific condition step
            n_1_m3 = conditions['pg_Pa'] / (k_B * conditions['Tg_K'])
            n_1_cm3 = n_1_m3 * 1e-6
            conditions['n_1_cm3'] = n_1_cm3

            print(f"\nEvaluating: Te={conditions['Te_eV']} eV, ne={conditions['n_e_cm3']:.1e} cm^-3")

            pops = self.run(
                Te_eV=conditions['Te_eV'],
                Tg_K=conditions['Tg_K'],
                n_e_cm3=conditions['n_e_cm3'],
                n_1_cm3=conditions['n_1_cm3'],
                n_ion_cm3=conditions['n_ion_cm3'],
                R_cm=conditions['R_cm'],
                analyze_levels=analyze_levels
            )
            all_results.append({'parameters': conditions, 'populations': pops})

        # Extract experimental data if it was provided in the TOML for plotting
        raw_exp_data = self.run_config.get("simulation", {}).get("experimental_data", {})
        experimental_data = {int(k): v for k, v in raw_exp_data.items()} if raw_exp_data else None

        # --- Combined Plotting & Exporting ---
        print("\n=== All Simulations Complete. Processing Outputs ===")

        # Plotting
        plotter = Plotter(database=self.db, output_dir=self.output_dir)
        show_blocks = self.run_config.get("plotting", {}).get("show_energy_blocks", False)
        plotter.plot_boltzmann(all_results, exp_data=experimental_data, is_optimization=False,
                               show_blocks=show_blocks)

        # Exporting
        if self.run_config.get("execution", {}).get("export_csv", True):
            exporter = Exporter(database=self.db, output_dir=self.output_dir)
            exporter.export_simulation(all_results)

    def _execute_optimization(self):
        """Handles the setup and execution of the optimization inverse problem."""
        print("=== Starting Optimization Mode ===")

        opt_cfg = self.run_config["optimization"]
        plasma_cfg = self.run_config["plasma"]
        dens_cfg = plasma_cfg["density"]

        # Parse experimental data (convert TOML string keys to ints)
        raw_exp_data = opt_cfg.get("experimental_data", {})
        experimental_data = {int(k): v for k, v in raw_exp_data.items()}

        # Extract initial guesses (assumes scalars here)
        Te_guess = plasma_cfg['Te_eV']
        Tg_K = plasma_cfg['Tg_K']
        pg_Pa = plasma_cfg['pg_Pa']
        R_cm = plasma_cfg['R_cm']
        n_e_cm3 = dens_cfg['n_e_cm3']
        n_ion_cm3 = dens_cfg['n_ion_cm3']

        n_1_cm3 = (pg_Pa / (k_B * Tg_K)) * 1e-6

        # Run optimizer
        best_Te, best_ne, errors, opt_pops = self.find_plasma_parameters(
            experimental_data=experimental_data,
            Te_guess=Te_guess,
            Tg_K=Tg_K,
            n_e_cm3=n_e_cm3,
            n_1_cm3=n_1_cm3,
            n_ion_cm3=n_ion_cm3,
            R_cm=R_cm,
            optimize_ne=opt_cfg.get("optimize_ne", False),
            Te_bounds=tuple(opt_cfg.get("Te_bounds", [0.1, 10.0])),
            ne_bounds=tuple(opt_cfg.get("ne_bounds", [1e10, 1e16])),
            analyze_optimal_levels=self.model_config.get("levels", {}).get("analyze")
        )

        # Pack optimized parameters for plotter
        optimal_params = {
            'Te_eV': best_Te, 'Tg_K': Tg_K, 'n_e_cm3': best_ne, 'pg_Pa': pg_Pa, 'R_cm': R_cm
        }

        # Pack optimized parameters for plotter
        optimal_params = {
            'Te_eV': best_Te, 'Tg_K': Tg_K, 'n_e_cm3': best_ne, 'pg_Pa': pg_Pa, 'R_cm': R_cm
        }
        opt_results = [{'parameters': optimal_params, 'populations': opt_pops}]

        # ---------------------------------------------------------
        # --- Combined Plotting & Exporting ---
        # ---------------------------------------------------------
        print("\n=== Optimization Complete. Processing Outputs ===")

        # Map errors BEFORE creating the results dictionary
        errors_dict = {'Te_eV': errors[0]}
        if opt_cfg.get("optimize_ne", False):
            errors_dict['n_e_cm3'] = errors[1]

        # Pack optimized parameters AND errors for the plotter
        optimal_params = {
            'Te_eV': best_Te, 'Tg_K': Tg_K, 'n_e_cm3': best_ne, 'pg_Pa': pg_Pa, 'R_cm': R_cm
        }
        opt_results = [{'parameters': optimal_params, 'errors': errors_dict, 'populations': opt_pops}]

        # Plotting
        plotter = Plotter(database=self.db, output_dir=self.output_dir)
        show_blocks = self.run_config.get("plotting", {}).get("show_energy_blocks", False)
        plotter.plot_boltzmann(opt_results, exp_data=experimental_data, is_optimization=True, show_blocks=show_blocks)

        # Exporting
        if self.run_config.get("execution", {}).get("export_csv", True):
            exporter = Exporter(database=self.db, output_dir=self.output_dir)

            # Map initial guesses and fixed parameters
            input_params = {
                'Te_eV': Te_guess, 'Tg_K': Tg_K, 'n_e_cm3': n_e_cm3, 'pg_Pa': pg_Pa, 'R_cm': R_cm
            }

            exporter.export_optimization(
                input_params=input_params,
                optimized_params=optimal_params,
                errors=errors_dict,
                optimal_pops=opt_pops
            )

    # =========================================================================
    # CORE METHODS (Kept mostly intact for manual usage)
    # =========================================================================

    def run(self, Te_eV, Tg_K, n_e_cm3, n_1_cm3, n_ion_cm3, R_cm, analyze_levels=None):
        """ Executes a steady-state CR simulation for a single state. """
        start_time = time.time()
        eedf = EEDF(Te_eV=Te_eV, mode=self.eedf_mode)

        populations = self.physics.build_and_solve(
            Te_eV=Te_eV, Tg_K=Tg_K, n_e=n_e_cm3, n_1=n_1_cm3,
            n_ion=n_ion_cm3, R_cm=R_cm, eedf_func=eedf
        )
        elapsed = time.time() - start_time
        print(f"Solved steady-state in {elapsed:.3f} seconds.")

        if analyze_levels:
            debugger = FluxAnalyzer(
                engine=self.physics, populations=populations, Te_eV=Te_eV, Tg_K=Tg_K,
                n_e=n_e_cm3, n_1=n_1_cm3, R_cm=R_cm, eedf_func=eedf
            )
            for lvl in analyze_levels:
                debugger.print_report(p=lvl)
        return populations

    # =========================================================================
    # PARAMETER OPTIMIZATION (INVERSE PROBLEM)
    # =========================================================================

    def find_plasma_parameters(self, experimental_data, Te_guess, Tg_K, n_e_cm3, n_1_cm3, n_ion_cm3, R_cm,
                               optimize_ne=False,
                               Te_bounds=(0.1, 10.0),
                               ne_bounds=(1e10, 1e16),
                               analyze_optimal_levels=None):
        """
        Finds the plasma parameters (Te and optionally n_e) that best match
        provided experimental population data.

        Uses a weighted least-squares optimization on logarithmic population ratios
        to ensure physical consistency across the entire level manifold.

        Parameters:
            experimental_data (dict): Experimental densities {level_id: {'density': float, 'weight': float}}.
            Te_guess (float): Starting guess for electron temperature in eV.
            Tg_K (float): Gas temperature in K.
            n_e_cm3 (float): Initial guess for electron density in cm^-3.
            n_1_cm3 (float): Ground state density in cm^-3.
            n_ion_cm3 (float): Ion density in cm^-3.
            R_cm (float): Plasma radius in cm.
            optimize_ne (bool): If True, both Te and n_e are optimized.
            Te_bounds (tuple): (min, max) bounds for Te.
            ne_bounds (tuple): (min, max) bounds for n_e.
            analyze_optimal_levels (list, optional): Levels to analyze after convergence.

        Returns:
            tuple: (best_Te, best_ne, errors, optimal_populations)
        """

        print(f"\nStarting {'2D (Te + ne)' if optimize_ne else '1D (Te only)'} Weighted Optimization.")

        # Configure optimization variables and bounds
        if optimize_ne:
            x0 = np.array([Te_guess, np.log10(n_e_cm3)], dtype=np.float64)
            limit_bounds = Bounds([Te_bounds[0], np.log10(ne_bounds[0])],
                                  [Te_bounds[1], np.log10(ne_bounds[1])])
        else:
            x0 = np.array([Te_guess], dtype=np.float64)
            limit_bounds = Bounds([Te_bounds[0]], [Te_bounds[1]])

        # Pre-process experimental data for faster evaluation
        exp_log_pops = {}
        weights = {}

        for lvl, data in experimental_data.items():
            exp_log_pops[lvl] = np.log(max(data['density'], 1e-99))
            weights[lvl] = data.get('weight', 1.0)

        levels = list(experimental_data.keys())
        pairs = list(combinations(levels, 2))

        def cost_function(x_array: np.ndarray) -> float:
            """Evaluates the difference between experimental and simulated population ratios."""
            current_Te = float(x_array[0])
            current_ne = 10 ** float(x_array[1]) if optimize_ne else n_e_cm3

            # Suppress diagnostics during optimization loop
            populations = self.run(
                Te_eV=current_Te, Tg_K=Tg_K, n_e_cm3=current_ne,
                n_1_cm3=n_1_cm3, n_ion_cm3=n_ion_cm3, R_cm=R_cm
            )

            total_error = 0.0
            for num_lvl, den_lvl in pairs:
                sim_num = max(populations.get(num_lvl, 1e-99), 1e-99)
                sim_den = max(populations.get(den_lvl, 1e-99), 1e-99)

                exp_log_ratio = exp_log_pops[num_lvl] - exp_log_pops[den_lvl]
                sim_log_ratio = np.log(sim_num) - np.log(sim_den)

                # Weight is the product of individual level weights
                pair_weight = weights[num_lvl] * weights[den_lvl]
                total_error += pair_weight * (exp_log_ratio - sim_log_ratio) ** 2

            return total_error

        # Execute optimization using L-BFGS-B (supports bounds)
        result = minimize(
            fun=cost_function,
            x0=x0,
            method='L-BFGS-B',
            bounds=limit_bounds,
            options={
                'ftol': 1e-4,
                'eps': 1e-2
            }
        )

        te_error, ne_error = 0.0, 0.0

        if result.success:
            best_Te = float(result.x[0])
            best_ne = 10 ** float(result.x[1]) if optimize_ne else float(n_e_cm3)
            try:
                # Estimate errors from the inverse Hessian (if available)
                variance_matrix = result.hess_inv.todense()
                te_error = float(np.sqrt(variance_matrix[0, 0]))
                if optimize_ne:
                    log_ne_error = float(np.sqrt(variance_matrix[1, 1]))
                    ne_error = best_ne * (10 ** log_ne_error - 1)
            except Exception:
                pass
            print(f"Optimization Converged!")
            print(f"Optimized Te: {best_Te:.4f} ± {te_error:.4f} eV")
        else:
            print(f"Optimization Failed: {result.message}")
            best_Te = float(result.x[0])
            best_ne = 10 ** float(result.x[1]) if optimize_ne else float(n_e_cm3)
            print(f"Returning closest fit: Te = {best_Te:.4f} eV")

        # Perform a final simulation run with optimal parameters to get final populations
        # and trigger any requested diagnostics.
        optimal_populations = self.run(
            Te_eV=best_Te, Tg_K=Tg_K, n_e_cm3=best_ne,
            n_1_cm3=n_1_cm3, n_ion_cm3=n_ion_cm3, R_cm=R_cm,
            analyze_levels=analyze_optimal_levels
        )

        return best_Te, best_ne, (te_error, ne_error), optimal_populations