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

import csv
from pathlib import Path
from datetime import datetime


class Exporter:
    """
    Handles data export for the Collisional-Radiative model.
    """

    def __init__(self, database, output_dir="output"):
        """
        Initializes the Exporter with a database reference and an output directory.

        Args:
            database (Database): Instance of the Database class for level metadata.
            output_dir (str or Path): Path where CSV files will be exported.
        """
        self.db = database
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_timestamp():
        """
        Generates a standardized timestamp string for unique file naming.

        Returns:
            str: Current timestamp in 'YYYY-MM-DD-HHMMSS' format.
        """
        return datetime.now().strftime("%Y-%m-%d-%H%M%S")

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

    def _write_population_data(self, file_handle, populations):
        """Standardized helper to write the exact same table structure to any file."""
        writer = csv.writer(file_handle)
        writer.writerow(["Level", "Density/cm^-3", "Excitation_Energy/eV", "Reduced_Density/cm^-3"])

        for lvl, density in sorted(populations.items()):
            try:
                lvl_data = self.db.query('levels', lower_level=lvl)
                g_m = lvl_data.get('g', 1.0)
                e_m = lvl_data.get('excitation_energy', 0.0)
            except (KeyError, ValueError):
                g_m, e_m = 1.0, 0.0

            reduced_density = density / g_m
            writer.writerow([lvl, f"{density:.6e}", f"{e_m:.4f}", f"{reduced_density:.6e}"])

    # =========================================================================
    # CORE EXPORT LOGIC
    # =========================================================================

    def export_simulation(self, results):
        """
        Exports simulation results. Automatically handles single runs vs. parameter sweeps.
        """
        timestamp = self._get_timestamp()

        # CASE 1: Single Simulation
        if len(results) == 1:
            filepath = self.output_dir / f"{timestamp}_simulation.csv"
            params = results[0]['parameters']
            pops = results[0]['populations']

            with open(filepath, mode='w', newline='', encoding='utf-8') as f:
                f.write("# ==========================================\n")
                f.write("# CR-Model Export: Single Simulation\n")
                f.write("# ==========================================\n")
                f.write("# --- Input Parameters ---\n")
                for k, v in params.items():
                    f.write(f"# {k}: {v:g}\n")
                f.write("#\n")
                self._write_population_data(f, pops)

            print(f"Saved simulation to: {filepath}")

        # CASE 2: Parameter Sweep
        else:
            varied_params, fixed_params = self._categorize_parameters(results)

            # Create a dedicated sweep directory
            sweep_dir = self.output_dir / f"{timestamp}_sweep"
            sweep_dir.mkdir(parents=True, exist_ok=True)

            # Determine sweep ranges for the header
            ranges = {}
            for k in varied_params:
                vals = [r['parameters'][k] for r in results]
                ranges[k] = (min(vals), max(vals))

            for res in results:
                params = res['parameters']
                pops = res['populations']

                # Create filename based on the varied parameters (e.g., Te_eV_3.0_pg_Pa_101325.csv)
                name_parts = [f"{k}_{params[k]:g}" for k in varied_params]
                filename = "_".join(name_parts) + ".csv"
                filepath = sweep_dir / filename

                with open(filepath, mode='w', newline='', encoding='utf-8') as f:
                    f.write("# ==========================================\n")
                    f.write("# CR-Model Export: Parameter Sweep\n")
                    f.write("# ==========================================\n")

                    f.write("# --- Sweep Ranges ---\n")
                    for k in varied_params:
                        min_v, max_v = ranges[k]
                        f.write(f"# {k}: {min_v:g} to {max_v:g}\n")

                    f.write("#\n# --- Fixed Parameters ---\n")
                    for k, v in fixed_params.items():
                        f.write(f"# {k}: {v:g}\n")

                    f.write("#\n# --- Current Run Parameters ---\n")
                    for k in varied_params:
                        f.write(f"# {k}: {params[k]:g}\n")
                    f.write("#\n")

                    self._write_population_data(f, pops)

            print(f"Saved {len(results)} sweep files to directory: {sweep_dir}")

    def export_optimization(self, input_params, optimized_params, errors, optimal_pops):
        """
        Exports the optimal populations of a parameter optimization run,
        including initial guesses and final errors in the header.
        """
        timestamp = self._get_timestamp()
        filepath = self.output_dir / f"{timestamp}_optimization.csv"

        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            f.write("# ==========================================\n")
            f.write("# CR-Model Export: Optimization\n")
            f.write("# ==========================================\n")

            f.write("# --- Initial Guesses & Fixed Parameters ---\n")
            for k, v in input_params.items():
                f.write(f"# {k}: {v:g}\n")

            f.write("#\n# --- Optimized Parameters ---\n")
            for k, v in optimized_params.items():
                err = errors.get(k, 0.0)
                f.write(f"# {k}: {v:g} +/- {err:g}\n")
            f.write("#\n")

            self._write_population_data(f, optimal_pops)

        print(f"Saved optimization results to: {filepath}")