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

from .constants import h, c, e


class SyntheticOES:
    """
    Translates Collisional-Radiative populations into theoretical optical emission spectra.
    
    This class uses the Einstein A coefficients and the calculated state populations 
    to determine the intensity of emission lines, effectively simulating what a 
    spectrometer would measure.
    """

    def __init__(self, database):
        """
        Initializes the synthetic OES generator.

        Parameters:
            database (Database): Instance of the database to fetch energy levels and 
                                 Einstein coefficients.
        """
        self.db = database

    def get_transition_wavelength_nm(self, upper, lower):
        """
        Calculates the theoretical vacuum wavelength for a transition between two levels.

        Parameters:
            upper (int): ID of the upper energy level.
            lower (int): ID of the lower energy level.

        Returns:
            float: Wavelength in nanometers (nm).
        """
        lvl_upper = self.db.query('levels', lower_level=upper)
        lvl_lower = self.db.query('levels', lower_level=lower)

        # Photon energy is the difference in excitation energies
        delta_E_eV = lvl_upper['excitation_energy'] - lvl_lower['excitation_energy']
        if delta_E_eV <= 0:
            return 0.0

        # Wavelength lambda = (h * c) / E
        delta_E_joules = delta_E_eV * e
        wavelength_m = (h * c) / delta_E_joules
        return wavelength_m * 1e9

    # =========================================================================
    # CORE CALCULATION LOGIC
    # =========================================================================

    def calculate_line_intensities(self, populations, target_transitions=None, wl_range=(200.0, 950.0)):
        """
        Calculates theoretical emission intensities for a set of transitions.

        Parameters:
            populations (dict): {level_id: density} from simulation.
            target_transitions (list, optional): List of (upper, lower) tuples to evaluate. 
                                                 If None, evaluates all possible transitions.
            wl_range (tuple): (min, max) wavelength range in nm to include in output.

        Returns:
            dict: Mapping of (upper, lower) -> {'wavelength_nm': float, 'intensity': float}.
                  Intensity is in units of photons / (cm^3 * s).
        """
        intensities = {}

        # Default to full range if None is passed
        wl_min, wl_max = wl_range if wl_range else (0.0, float('inf'))

        # --- FAST PATH: Calculate only specifically requested lines (e.g., for optimizer) ---
        if target_transitions is not None:
            for (upper, lower) in target_transitions:
                n_p = populations.get(upper, 0.0)
                # Query emission coefficient A_pq (s^-1)
                A_pq = self.db.query('emission', lower_level=lower, upper_level=upper)

                if A_pq > 0:
                    wl = self.get_transition_wavelength_nm(upper, lower)
                    # Filter by wavelength range
                    if wl_min <= wl <= wl_max:
                        intensities[(upper, lower)] = {
                            'wavelength_nm': wl,
                            'intensity': n_p * A_pq
                        }
            return intensities

        # --- SLOW PATH: Full Spectrum Generation (loops over all level pairs) ---
        active_levels = list(populations.keys())
        for upper in active_levels:
            n_p = populations[upper]
            if n_p <= 0:
                continue

            # Iterate over all possible lower levels for this upper level
            for lower in range(2, upper):  # Note: starting at 2 skips deep UV resonance lines
                A_pq = self.db.query('emission', lower_level=lower, upper_level=upper)
                if A_pq > 0:
                    wl = self.get_transition_wavelength_nm(upper, lower)

                    # Hardware range filter
                    if wl_min <= wl <= wl_max:
                        intensities[(upper, lower)] = {
                            'wavelength_nm': wl,
                            'intensity': n_p * A_pq
                        }

        return intensities