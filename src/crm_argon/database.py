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
import pathlib
import numpy as np


class Database:
    """
    Atomic database for the Collisional-Radiative model.
    
    This class handles the loading, parsing, and querying of all atomic data 
    required for the Argon CR model, including energy levels, Einstein coefficients, 
    and cross-section parameters for excitation and ionization.
    """
    def __init__(self, custom_data_path=None):
        """
        Initializes the database and loads all CR model parameters from CSV files.

        Parameters:
            custom_data_path (str, optional): Custom path to the directory containing 
                                             the atomic data CSV files.
        """
        if custom_data_path is None:
            # Default to the 'data' directory relative to this script
            self.data_dir = pathlib.Path(__file__).parent / "data"
        else:
            self.data_dir = pathlib.Path(custom_data_path)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Database directory not found at: {self.data_dir}. "
                "Ensure the 'data' folder is inside the 'cr_model' package."
            )

        # Internal Data Storage
        self.levels = {}
        self.ionization = {}
        self.photoionization = {}
        self.metastable_quenching = {}

        # Dictionary mapping (lower, upper) -> [list of parameter dicts] for electron excitation
        self.excitation_params = {}

        # 65x65 Matrix for Einstein A coefficients (s^-1)
        self.emission_matrix = np.zeros((65, 65))

        # Sequentially load all atomic data components
        self._load_levels()
        self._load_emission()
        self._load_excitation_allowed()
        self._load_excitation_parity_forbidden()
        self._load_excitation_spin_forbidden()
        self._load_ionization()
        self._load_photoionization()
        self._load_metastable_quenching()

        print("Database initialized successfully.")

    # =========================================================================
    # DATA LOADING METHODS
    # =========================================================================

    def _read_csv_safely(self, filename):
        """
        Reads a CSV file and returns its content as a list of dictionaries.

        Parameters:
            filename (str): Name of the CSV file to load.

        Returns:
            list: List of dictionaries representing the CSV rows.
        """
        filepath = self.data_dir / filename

        try:
            with open(filepath, mode='r', encoding='utf-8') as f:
                return list(csv.DictReader(f))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"CRITICAL: Missing required data file '{filename}'. "
                f"Expected to find it at: {filepath}"
            ) from e

        except Exception as e:
            raise RuntimeError(f"Unexpected error while reading {filename}: {e}") from e

    def _parse_qn_field(self, value, is_int=False):
        """
        Parses quantum number fields that may contain multiple values separated by '/'.

        Parameters:
            value (str): The raw string from the CSV.
            is_int (bool): Whether to attempt casting the values to integers.

        Returns:
            list: A list of parsed quantum numbers.
        """
        if not value:
            return []

        parts = [p.strip() for p in str(value).split('/')]

        if is_int:
            parsed_parts = []
            for p in parts:
                try:
                    parsed_parts.append(int(p))
                except ValueError:
                    # Keep as string if parsing fails (e.g., 'all')
                    parsed_parts.append(p)
            return parsed_parts

        return parts

    def _load_levels(self):
        """Loads atomic level properties: energy, statistical weight, and quantum numbers."""
        data = self._read_csv_safely("excitation_levels.csv")

        try:
            for row in data:
                n = int(row['n'])
                energy = float(row.get('excitation_energy', row.get('energy_ev', 0.0)))

                self.levels[n] = {
                    'designation': row['designation'],
                    'excitation_energy': energy,
                    'g': float(row['g']) if row.get('g') else 1.0,
                    'qn_n': self._parse_qn_field(row.get('qn_n'), is_int=True),
                    'qn_l': self._parse_qn_field(row.get('qn_l')),
                    'qn_K': self._parse_qn_field(row.get('qn_K')),
                    'qn_J': self._parse_qn_field(row.get('qn_J')),
                    'qn_core': self._parse_qn_field(row.get('qn_core'))
                }
        except KeyError as e:
            raise ValueError(f"FORMAT ERROR: Missing expected column {e} in 'excitation_levels.csv'.") from e

    def _add_excitation(self, lower, upper, param_dict):
        """Registers excitation parameters for a specific level transition."""
        if (lower, upper) not in self.excitation_params:
            self.excitation_params[(lower, upper)] = []
        self.excitation_params[(lower, upper)].append(param_dict)

    def _load_emission(self):
        """Loads spontaneous emission coefficients (Einstein A) into the matrix."""
        data = self._read_csv_safely("einstein_parameters.csv")
        try:
            for row in data:
                upper, lower = int(row['upper_n']), int(row['lower_n'])
                self.emission_matrix[upper - 1, lower - 1] = float(row['A'])
        except KeyError as e:
            raise ValueError(f"FORMAT ERROR: Missing expected column {e} in 'einstein_params.csv'.") from e

    def _load_excitation_allowed(self):
        """Loads parameters for optically allowed (A) electron impact transitions."""
        data = self._read_csv_safely("excitation_allowed.csv")
        try:
            for row in data:
                m, n = int(row['lower_n']), int(row['upper_n'])
                self._add_excitation(m, n, {
                    'type': 'A',
                    'excitation_energy': float(row['excitation_energy']),
                    'alpha': float(row['alpha']),
                    'beta': float(row['beta'])
                })
        except KeyError as e:
            raise ValueError(f"FORMAT ERROR: Missing expected column {e} in 'excitation_allowed.csv'.") from e

    def _load_excitation_parity_forbidden(self):
        """Loads parameters for parity-forbidden (P) electron impact transitions."""
        data = self._read_csv_safely("excitation_parity_forbidden.csv")
        try:
            for row in data:
                m, n = int(row['lower_n']), int(row['upper_n'])
                self._add_excitation(m, n, {
                    'type': 'P',
                    'excitation_energy': float(row['excitation_energy']),
                    'alpha': float(row['alpha'])
                })
        except KeyError as e:
            raise ValueError(f"FORMAT ERROR: Missing expected column {e} in 'excitation_parity_forbidden.csv'.") from e

    def _load_excitation_spin_forbidden(self):
        """Loads parameters for spin-forbidden (S) electron impact transitions."""
        data = self._read_csv_safely("excitation_spin_forbidden.csv")
        try:
            for row in data:
                m, n = int(row['lower_n']), int(row['upper_n'])
                self._add_excitation(m, n, {
                    'type': 'S',
                    'excitation_energy': float(row['excitation_energy']),
                    'alpha': float(row['alpha']),
                    'equation': int(row['equation'])
                })
        except KeyError as e:
            raise ValueError(f"FORMAT ERROR: Missing expected column {e} in 'excitation_spin_forbidden.csv'.") from e

    def _load_ionization(self):
        """Loads electron impact ionization parameters for neutral levels."""
        data = self._read_csv_safely("ionization.csv")
        try:
            for row in data:
                n = int(row['level_n'])
                self.ionization[n] = {
                    'ionization_energy': float(row['ionization_energy']),
                    'xi_ion': float(row['xi_ion']) if row['xi_ion'] else None,
                    'alpha_ion': float(row['alpha_ion']) if row['alpha_ion'] else None,
                    'beta_ion': float(row['beta_ion']) if row['beta_ion'] else None
                }
        except KeyError as e:
            raise ValueError(f"FORMAT ERROR: Missing expected column {e} in 'ionization.csv'.") from e

    def _load_photoionization(self):
        """Loads photoionization correction factors used for radiative recombination."""
        data = self._read_csv_safely("photoionization.csv")
        try:
            for row in data:
                n = int(row['n'])
                self.photoionization[n] = {
                    'gamma_p': float(row['gamma_p'])
                }
        except KeyError as e:
            raise ValueError(f"FORMAT ERROR: Missing expected column {e} in 'photoionization.csv'.") from e

    def _load_metastable_quenching(self):
        """Loads heavy-particle quenching constants for metastable states."""
        data = self._read_csv_safely("metastable_quenching.csv")
        try:
            for row in data:
                n = int(row['level_n'])
                self.metastable_quenching[n] = {
                    'd_n': float(row['d_n']),
                    'B_n': float(row['B_n'])
                }
        except KeyError as e:
            raise ValueError(f"FORMAT ERROR: Missing expected column {e} in 'metastable_quenching.csv'.") from e

    # =========================================================================
    # DATA QUERY METHODS
    # =========================================================================

    def _validate_level(self, level):
        """Ensures the queried level index is within the valid range."""
        if level not in self.levels:
            max_lvl = max(self.levels.keys()) if self.levels else 65
            raise ValueError(f"Level {level} is out of bounds. Must be between 1 and {max_lvl}.")

    def get_emission_data(self, lower_level=None, upper_level=None):
        """Retrieves Einstein A coefficients."""
        if lower_level is None and upper_level is None:
            return self.emission_matrix
        elif lower_level is not None and upper_level is not None:
            self._validate_level(lower_level)
            self._validate_level(upper_level)
            return self.emission_matrix[upper_level - 1, lower_level - 1]
        else:
            raise ValueError("For 'emission', you must specify BOTH lower_level and upper_level, or NEITHER.")

    def get_excitation_data(self, lower_level, upper_level):
        """Retrieves electron excitation parameters for a specific level pair."""
        if lower_level is None or upper_level is None:
            raise ValueError("For excitation data, you must specify BOTH lower_level and upper_level.")
        self._validate_level(lower_level)
        self._validate_level(upper_level)
        return self.excitation_params.get((lower_level, upper_level), [])

    def get_ionization_data(self, level):
        """Retrieves electron ionization parameters for a level."""
        self._validate_level(level)
        return self.ionization.get(level, None)

    def get_photoionization_data(self, level):
        """Retrieves photoionization parameters for a level."""
        self._validate_level(level)
        return self.photoionization.get(level, None)

    def get_level_data(self, level):
        """Retrieves basic properties (energy, g) for a level."""
        self._validate_level(level)
        return self.levels.get(level, None)

    def get_metastable_quenching_data(self, level):
        """Retrieves quenching constants for a level (returns zeros for non-metastables)."""
        self._validate_level(level)
        return self.metastable_quenching.get(level, {'d_n': 0.0, 'B_n': 0.0})

    # ==========================================
    # UNIFIED QUERY INTERFACE
    # ==========================================

    def query(self, data_type, lower_level=None, upper_level=None):
        """
        Unified entry point to query atomic data for the CR model.

        Parameters:
            data_type (str): Type of data to query (e.g., 'levels', 'emission', 'excitation').
            lower_level (int, optional): ID of the lower level.
            upper_level (int, optional): ID of the upper level.

        Returns:
            Various: Requested data in suitable format (dict, float, or array).
        """
        data_type = data_type.lower()

        # Sanity checks for level IDs
        if lower_level and upper_level and lower_level > upper_level:
            raise ValueError("Lower level cannot be greater than upper level.")

        max_n = max(self.levels.keys()) if self.levels else 65
        if lower_level and lower_level > max_n:
            raise ValueError(f"Lower level index {lower_level} exceeds database maximum {max_n}.")

        if upper_level and upper_level > max_n:
            raise ValueError(f"Upper level index {upper_level} exceeds database maximum {max_n}.")

        # Routing the query to the specific getter method
        if data_type == 'emission':
            return self.get_emission_data(lower_level, upper_level)

        if data_type == 'excitation':
            return self.get_excitation_data(lower_level, upper_level)

        if data_type in ['levels', 'ionization', 'photoionization', 'metastable_quenching']:
            if lower_level is None:
                raise ValueError(f"For '{data_type}', you must specify at least 'lower_level'.")

            if data_type == 'levels':
                return self.get_level_data(lower_level)
            elif data_type == 'ionization':
                return self.get_ionization_data(lower_level)
            elif data_type == 'photoionization':
                return self.get_photoionization_data(lower_level)
            elif data_type == 'metastable_quenching':
                return self.get_metastable_quenching_data(lower_level)

        raise ValueError(f"Unknown data_type '{data_type}'. Valid options: "
                         f"levels, emission, excitation, ionization, photoionization, metastable_quenching.")