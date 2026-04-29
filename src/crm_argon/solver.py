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


class SteadyStateSolver:
    """
    Solves the Quasi-Stationary State (QSS) linear system for excited state populations.
    Executes the linear algebra equivalent of Vlcek (1989) Equation (1).
    """

    def __init__(self):
        pass

    def build_rhs_vector(self, delta_m, a_m1, n1):
        """
        Helper method to construct the exact right-hand side (b) vector
        according to Vlcek Eq. (1): -delta_m - (a_m1 * n1)
        
        The right-hand side represents source terms from the ground state and 
        recombination that are constant with respect to the excited state populations.

        Parameters:
        - delta_m: Recombination source terms into level m [m^-3 s^-1]
        - a_m1: Excitation rate from ground state into level m [s^-1]
        - n1: Population density of the ground state [m^-3]

        Returns:
        - b_vector: 1D numpy array of length N
        """
        return -np.array(delta_m) - (np.array(a_m1) * n1)

    # =========================================================================
    # MATRIX SOLUTION (A * x = b)
    # =========================================================================

    def solve(self, A_matrix, b_vector):
        """
        Solves the linear system A * x = b for the population array x.

        Parameters:
        - A_matrix: 2D numpy array of shape (N, N) representing the collisional-radiative rate matrix.
                    (Typically 64x64, excluding the ground state).
        - b_vector: 1D numpy array of shape (N) representing the source terms.

        Returns:
        - n_array: 1D numpy array of shape (N) containing the calculated excited state populations.
        """
        # Ensure inputs are standard numpy arrays
        A = np.asarray(A_matrix, dtype=float)
        b = np.asarray(b_vector, dtype=float)

        # Dimension validation
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix A must be square. Received shape {A.shape}.")
        if A.shape[0] != b.shape[0]:
            raise ValueError(f"Matrix A dimension ({A.shape[0]}) must match vector b dimension ({b.shape[0]}).")

        try:
            # np.linalg.solve is highly optimized for exactly this type of problem
            n_array = np.linalg.solve(A, b)

            # Physical validity check: Populations cannot be negative
            # A well-conditioned CR matrix (negative diagonals, positive off-diagonals)
            # should naturally yield positive populations.
            if np.any(n_array < 0):
                print("WARNING: Solver yielded negative populations. "
                      "Check your rate matrix (A) for physical consistency. "
                      "Ensure diagonal loss terms (Eq 4) are correctly populated and negative.")

            return n_array

        except np.linalg.LinAlgError as e:
            # This triggers if the matrix is singular (determinant = 0), which usually means
            # the plasma conditions are completely unphysical or a rate evaluated to NaN/Infinity.
            raise RuntimeError(f"Linear algebra solver failed: {e}. Matrix may be singular or poorly conditioned.")