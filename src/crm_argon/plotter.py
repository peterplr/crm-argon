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

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


class Plotter:
    """
    Handles all data visualization and plotting for the Collisional-Radiative model.
    """

    def __init__(self, database, output_dir):
        """
        Initializes the Plotter with a database reference and an output directory.

        Args:
            database (Database): Instance of the Database class for level metadata.
            output_dir (str or Path): Path to the directory where plots will be saved.
        """
        self.db = database
        self.output_dir = output_dir

        # Parameter formatting dictionary for pretty labels
        self.param_labels = {
            'Te_eV': ('$T_e$', 'eV'),
            'Tg_K': ('$T_g$', 'K'),
            'n_e_cm3': ('$n_e$', 'cm$^{-3}$'),
            'pg_Pa': ('$p_g$', 'Pa'),
            'R_cm': ('$R$', 'cm'),
            'n_1_cm3': ('$n_1$', 'cm$^{-3}$')
        }

    @staticmethod
    def _get_timestamp():
        """
        Generates a standardized timestamp string for unique file naming.

        Returns:
            str: Current timestamp in 'YYYY-MM-DD-HHMMSS' format.
        """
        return datetime.now().strftime("%Y-%m-%d-%H%M%S")

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def plot_boltzmann(self, results, exp_data=None, is_optimization=False, show_blocks=False):
        """
        Generates a Boltzmann plot (reduced density vs. excitation energy).
        Can overlay experimental data and highlight energy level blocks.

        Args:
            results (list): List of simulation result dictionaries.
            exp_data (dict, optional): Experimental data to overlay.
            is_optimization (bool): If True, treats the result as an optimization outcome.
            show_blocks (bool): If True, adds color-coded blocks for 4s, 4p, etc.
        """
        # Load global style if available
        style_path = Path(__file__).parent.parent.parent / "configs" / "crm_argon.mplstyle"
        if style_path.exists():
            plt.style.use(str(style_path))
        else:
            print("Warning: crm_argon.mplstyle not found. Using default styles.")

        print("\nGenerating Boltzmann plot...")

        # Setup figure and axes
        fig, ax = plt.subplots(figsize=(6,5))

        varied_params, fixed_params = self._categorize_parameters(results)

        # Extract errors if they were provided (typically only in optimization)
        errors = results[0].get('errors', {}) if len(results) == 1 else {}

        self._plot_simulation_data(ax, results, varied_params)

        if exp_data:
            target_populations = results[0]['populations']
            self._plot_experimental_data(ax, target_populations, exp_data)

        if show_blocks:
            self._add_energy_blocks(ax)

        self._format_axes(ax)

        # Pass the errors dictionary to the header
        self._add_parameter_header(ax, fixed_params, errors)
        self._save_figure(fig, ax, is_optimization)

    # =========================================================================
    # MODULAR HELPERS
    # =========================================================================

    @staticmethod
    def _categorize_parameters(results):
        """
        Identifies which parameters varied across runs and which stayed constant.
        Intelligently handles dependent physics variables.
        """
        if len(results) == 1:
            return [], results[0]['parameters']

        all_keys = list(results[0]['parameters'].keys())
        varied = []
        fixed = {}

        for key in all_keys:
            values = [res['parameters'][key] for res in results]

            if len(set(values)) > 1:
                varied.append(key)
            else:
                fixed[key] = values[0]

        # --- PHYSICS DEPENDENCY CHECK ---
        # n_1_cm3 is functionally dependent on Tg_K and pg_Pa.
        # If either T or p varies, n_1 naturally varies. We remove it from the
        # varied list so it doesn't clutter legends or filenames, unless it was
        # the ONLY thing that varied.
        if 'n_1_cm3' in varied and ('Tg_K' in varied or 'pg_Pa' in varied):
            varied.remove('n_1_cm3')

        return varied, fixed

    def _extract_plot_data(self, populations):
        """Extracts energies and reduced densities, skipping the ground state."""
        energies, reduced_densities = [], []

        for lvl, n_m in populations.items():
            if lvl < 2:
                continue

            lvl_data = self.db.query('levels', lower_level=lvl)
            g_m = lvl_data['g']
            E_m = lvl_data['excitation_energy']

            energies.append(E_m)
            reduced_densities.append(n_m / g_m)

        return energies, reduced_densities

    @staticmethod
    def _format_value(value):
        """
        Formats numbers into standard string representation or LaTeX scientific notation,
        restricted to 4 significant digits.
        """
        if value == 0:
            return "$0$"

        # Standard numbers (formatted to 4 significant digits using .4g)
        if 0.01 <= abs(value) < 10000:
            return f"${value:.4g}$"

        # For large/small numbers, convert to scientific notation (.3e gives 4 sig figs total)
        base, exponent = f"{value:.3e}".split('e')
        exp_int = int(exponent)

        # Clean up the base (removes trailing zeros to keep it clean)
        base = base.rstrip('0').rstrip('.')

        # Special case: don't write "1 \cdot 10^5", just write "10^5"
        if base == "1" or base == "-1":
            sign = "-" if base == "-1" else ""
            return f"${sign}10^{{{exp_int}}}$"
        else:
            return f"${base} \\cdot 10^{{{exp_int}}}$"

    @staticmethod
    def _format_with_prefix(value, unit):
        """
        Converts a value to Engineering Notation with SI prefixes,
        restricted to 4 significant digits.
        """
        import math
        if value == 0:
            return "$0$", unit

        # Standard SI Prefix dictionary (uses Unicode 'μ' for micro)
        prefixes = {
            24: 'Y', 21: 'Z', 18: 'E', 15: 'P', 12: 'T', 9: 'G', 6: 'M', 3: 'k',
            0: '',
            -3: 'm', -6: 'μ', -9: 'n', -12: 'p', -15: 'f'
        }

        # Find the nearest power of 3
        magnitude = math.floor(math.log10(abs(value)) / 3) * 3

        # Constrain to our dictionary bounds
        if magnitude > 24:
            magnitude = 24
        elif magnitude < -15:
            magnitude = -15

        prefix = prefixes[magnitude]
        scaled_value = value / (10 ** magnitude)

        # Format the scaled value strictly to 4 significant digits using .4g
        val_str = f"${scaled_value:.4g}$"

        # Attach the prefix directly to the unit (e.g., 'k' + 'Pa' = 'kPa')
        combined_unit = f"{prefix}{unit}"

        return val_str, combined_unit

    def _format_value_with_error(self, value, error):
        r"""
        Formats a value and its error using LaTeX \pm notation.
        Automatically aligns the decimal precision of the value to match the error.
        """
        import math

        if error == 0:
            return self._format_value(value)

        # --- Standard numbers ---
        if 0.01 <= abs(value) < 10000:
            # Find the magnitude of the error to determine rounding (assuming 2 sig figs for the error)
            err_mag = math.floor(math.log10(abs(error)))
            round_place = err_mag - 1

            if round_place > 0:
                # For large integer errors (e.g., error = 150 -> round to tens)
                factor = 10 ** round_place
                v_rnd = int(round(value / factor) * factor)
                e_rnd = int(round(error / factor) * factor)
                return f"${v_rnd} \\pm {e_rnd}$"
            else:
                # For decimal errors (e.g., error = 0.15 -> round to 2 decimals)
                decimals = -round_place
                # 'f' formatting keeps the trailing zeros, which is physically correct!
                return f"${value:.{decimals}f} \\pm {error:.{decimals}f}$"

        # --- Scientific notation ---
        base_exp = math.floor(math.log10(abs(value))) if value != 0 else 0

        # Scale both the value and the error to fit in the same bracket (e.g., * 10^5)
        scaled_v = value / (10 ** base_exp)
        scaled_e = error / (10 ** base_exp)

        if scaled_e != 0:
            scaled_err_mag = math.floor(math.log10(abs(scaled_e)))
            round_place = scaled_err_mag - 1
            decimals = max(0, -round_place)

            v_str = f"{scaled_v:.{decimals}f}"
            e_str = f"{scaled_e:.{decimals}f}"
        else:
            v_str = f"{scaled_v:.3f}"
            e_str = "0"

        return f"$({v_str} \\pm {e_str}) \\cdot 10^{{{base_exp}}}$"

    def _plot_simulation_data(self, ax, results, varied_params):
        """Plots scatter data. Colors and base shapes are pulled from the style sheet.
           Forces the markers to be HOLLOW to distinguish from experimental data."""
        for i, res in enumerate(results):
            pops = res['populations']
            params = res['parameters']
            energies, reduced_densities = self._extract_plot_data(pops)

            if not varied_params:
                label = 'CR Model'
            else:
                label_parts = []
                for k in varied_params:
                    name, unit = self.param_labels.get(k, (k, ""))
                    # Use the new prefix formatter for the legend
                    val_str, combined_unit = self._format_with_prefix(params[k], unit)
                    label_parts.append(f"{name} = {val_str} {combined_unit}".strip())
                label = " | ".join(label_parts)

            ax.semilogy(energies, reduced_densities, linestyle='',
                        alpha=0.9, label=label, zorder=3,
                        markerfacecolor='none', markeredgewidth=1.5)

    def _plot_experimental_data(self, ax, sim_populations, exp_data):
        """Scales and plots experimental data against a simulation baseline.
           Safely handles both scalar inputs (simulation) and dict inputs (optimization)."""
        exp_E, exp_n_g_raw, sim_n_g_target = [], [], []

        for lvl, exp_val in exp_data.items():
            # Automatically extract the density depending on the dictionary format
            if isinstance(exp_val, dict):
                exp_dens = exp_val.get('density', 0.0)
            else:
                exp_dens = float(exp_val)

            # Skip missing or zero data
            if exp_dens <= 0:
                continue

            lvl_data = self.db.query('levels', lower_level=lvl)
            g_m = lvl_data['g']
            E_m = lvl_data['excitation_energy']

            exp_E.append(E_m)
            exp_n_g_raw.append(exp_dens / g_m)
            sim_n_g_target.append(sim_populations[lvl] / g_m)

        scale_factor = np.exp(np.mean(np.log(sim_n_g_target) - np.log(exp_n_g_raw)))
        exp_n_g_scaled = [val * scale_factor for val in exp_n_g_raw]

        # Explicitly define the SOLID red box ('s').
        ax.semilogy(exp_E, exp_n_g_scaled, marker='s', linestyle='', color='#e03a3c',
                    markeredgecolor='black', markeredgewidth=0.8, markersize=7,
                    label='Experimental', zorder=5)

    @staticmethod
    def _add_energy_blocks(ax):
        """Adds vertical shaded regions for characteristic Argon energy blocks."""
        ax.axvspan(11.5, 11.9, color='red', alpha=0.1, label='4s block', zorder=1)
        ax.axvspan(12.9, 13.5, color='green', alpha=0.1, label='4p block', zorder=1)
        ax.axvspan(13.8, 14.31, color='purple', alpha=0.1, label='3d + 5s block', zorder=1)
        ax.axvspan(14.5, 14.7, color='blue', alpha=0.1, label='5p block', zorder=1)

    @staticmethod
    def _format_axes(ax):
        """Applies labels. Grids, colors, and fonts are handled by the style sheet."""
        ax.set_xlabel("Excitation Energy $\\varepsilon_m$ / eV")
        ax.set_ylabel("Reduced Population Density $n_m / g_m$ / cm$^{-3}$")
        ax.legend(loc='best')

    def _add_parameter_header(self, ax, fixed_params, errors=None):
        """Constructs a clean title string split into two lines, including errors if available."""
        if errors is None:
            errors = {}

        header_parts = []
        for k, v in fixed_params.items():
            if k not in self.param_labels or k == 'n_1_cm3':
                continue

            name, unit = self.param_labels[k]

            # If an error exists for this parameter, use the \pm formatter
            if k in errors and errors[k] is not None and errors[k] > 0:
                val_str = self._format_value_with_error(v, errors[k])
            else:
                val_str = self._format_value(v)

            header_parts.append(f"{name} = {val_str} {unit}")

        title_text = "  |  ".join(header_parts)
        ax.set_title(title_text, pad=15)

    def _save_figure(self, fig, ax, is_optimization):
        """Saves the figure to disk and closes it to free memory."""
        # Adjust layout to accommodate the external legend
        fig.tight_layout()

        save_path = Path(self.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        timestamp = self._get_timestamp()
        suffix = "_boltzmann_opt" if is_optimization else "_boltzmann_sim"
        filename = f"{timestamp}{suffix}.png"

        full_save_path = save_path / filename
        fig.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved as '{full_save_path}'")

        # Crucial: Close the figure so it doesn't display or leak memory
        plt.close(fig)