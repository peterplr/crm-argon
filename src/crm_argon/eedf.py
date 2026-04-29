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
from scipy.special import gamma
from typing import Union
from .constants import m_e, e


class EEDF:
    """
    Calculates the Electron Energy Distribution Function (EEDF) f(u).
    
    Currently implements the Maxwellian distribution following the normalization 
    convention of Akatsuka (2009). The EEDF is defined in velocity phase-space.
    """

    def __init__(self, Te_eV: float, mode: str = 'maxwellian'):
        """
        Initializes the EEDF with a specific electron temperature and mode.

        Parameters:
            Te_eV (float): Electron temperature in eV.
            mode (str): Type of distribution ('maxwellian' is supported).
        """
        self.mode = mode.lower().strip()
        self.Te_eV = Te_eV

        if self.mode == 'maxwellian':
            self.g = 1.0  # Maxwellian exponent
        elif self.mode == 'druyvesteyn':
            # Druyvesteyn requires a different integration approach in the Kinetics engine
            raise NotImplementedError(
                "Druyvesteyn EEDF is currently not supported by the Kinetics engine. \n"
                "The current model relies on analytical detailed balance shortcuts "
                "that strictly assume a Maxwellian distribution. \n"
                "To use a Druyvesteyn distribution, the fundamental rate integrals "
                "in kinetics.py must be modified to use the full cross-sections."
            )
        else:
            raise ValueError("Mode must be strictly 'maxwellian' or 'druyvesteyn'.")

        self._calculate_coefficients()

    # =========================================================================
    # CORE LOGIC
    # =========================================================================

    def _calculate_coefficients(self):
        """
        Pre-calculates normalization constants c1 and c2 based on the mode (g).
        """
        gamma_25 = gamma(2.5 / self.g)
        gamma_15 = gamma(1.5 / self.g)

        # Standard normalization constants for Generalized Maxwellian distributions
        self.c2 = ((2.0 / 3.0) * (gamma_25 / gamma_15)) ** self.g
        self.c1 = self.g * (2.0 / 3.0) ** 1.5 * gamma_25 ** 1.5 / gamma_15 ** 2.5

        # Phase-space volume prefactor based on Akatsuka (2009)
        Te_joules = self.Te_eV * e
        self.akatsuka_norm = 2.0 * np.pi * ((2.0 * Te_joules) / m_e) ** 1.5

    def __call__(self, u: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluates the normalized EEDF at the dimensionless energy u = E / Te.

        Parameters:
            u (float or ndarray): Dimensionless electron energy.

        Returns:
            float or ndarray: The value of the EEDF at u.
        """
        u_arr = np.asarray(u)
        u_safe = np.where(u_arr > 0, u_arr, 0.0)

        # Evaluates f(u) = (c1/norm) * exp(-c2 * u^g)
        result = (self.c1 / self.akatsuka_norm) * np.exp(-self.c2 * (u_safe ** self.g))

        final_result = np.where(u_arr > 0, result, 0.0)
        return final_result.item() if np.isscalar(u) or u_arr.ndim == 0 else final_result