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

# =============================================================================
# Fundamental Physical Constants
# =============================================================================
h = 6.62607015e-34      # Planck's constant [J*s]
k_B = 1.380649e-23      # Boltzmann constant [J/K]
m_e = 9.10938356e-31    # Electron mass [kg]
e = 1.602176634e-19     # Elementary charge [C]
c = 299792458           # Speed of light [m/s]
a0 = 5.29177210545e-11  # First Bohr Radius [m]
a0_cm = a0 * 1e2        # First Bohr Radius [cm]
pi = 3.141592653589793  # Pi constant

# =============================================================================
# Conversion Factors
# =============================================================================
EV_TO_J = e                     # Conversion from eV to Joules
J_TO_EV = 1 / e                 # Conversion from Joules to eV
AMU_TO_KG = 1.6605390689e-27    # Atomic mass unit to kg

# =============================================================================
# Atomic Data (Argon & Hydrogen)
# =============================================================================
E_ION_H = 13.59844      # Ionization energy for Hydrogen [eV]
M_AR = 39.948           # Argon atomic mass [amu]
E_ION_AR = 15.7596119   # Ionization energy for Argon ground state (3p6) [eV]
