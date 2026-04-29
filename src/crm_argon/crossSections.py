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
from .constants import E_ION_H, a0_cm


class CrossSections:
    """
    Defines the cross sections for various atomic processes in Argon.
    
    This class implements analytical formulas for excitation and ionization cross sections.
    It uses 'function factories' (closures) to return functions that evaluate the 
    cross section at a given energy, which allows for pre-calculating constants and 
    improves performance during numerical integration.
    """

    def __init__(self):
        # 4 * pi * a0^2 constant used in many cross-section formulas [cm^2]
        self.four_pi_a02 = 4.0 * np.pi * (a0_cm ** 2)

    # ==========================================
    # ELECTRON COLLISIONS (Returns Functions)
    # ==========================================

    # =========================================================================
    # ELECTRON EXCITATION (VRIENS & SREETS)
    # =========================================================================

    def allowed(self, eps_mn, alpha_f_mn, beta_mn):
        """
        Formula for Optically Allowed Transitions (A).
        Based on Vlcek (1989) Eq (4).
        
        Parameters:
            eps_mn (float): Threshold energy in eV.
            alpha_f_mn (float): Fitting parameter alpha * f_mn.
            beta_mn (float): Fitting parameter beta.

        Returns:
            callable: A function sigma(E) returning the cross section in cm^2.
        """
        prefactor = self.four_pi_a02 * ((E_ION_H / eps_mn) ** 2) * alpha_f_mn

        def sigma(E):
            """Evaluates cross section at energy E [eV]."""
            if np.isscalar(E):
                if E <= eps_mn: return 0.0
                u = E / eps_mn
                return prefactor * (u ** -2) * (u - 1.0) * np.log(1.25 * beta_mn * u)
            else:
                E_arr = np.asarray(E)
                u = np.where(E_arr > eps_mn, E_arr / eps_mn, 1.0)  # Prevent zero div
                res = np.where(E_arr > eps_mn, prefactor * (u ** -2) * (u - 1.0) * np.log(1.25 * beta_mn * u), 0.0)
                return res

        return sigma

    def parity_forbidden(self, eps_mn, alpha_mn):
        """
        Formula for Parity-Forbidden Transitions (P).
        Based on Vlcek (1989) Eq (5).

        Parameters:
            eps_mn (float): Threshold energy in eV.
            alpha_mn (float): Fitting parameter.

        Returns:
            callable: A function sigma(E) in cm^2.
        """
        prefactor = self.four_pi_a02 * alpha_mn

        def sigma(E):
            if np.isscalar(E):
                if E <= eps_mn: return 0.0
                u = E / eps_mn
                return prefactor * (u ** -1) * (1.0 - (u ** -1))
            else:
                E_arr = np.asarray(E)
                u = np.where(E_arr > eps_mn, E_arr / eps_mn, 1.0)
                return np.where(E_arr > eps_mn, prefactor * (u ** -1) * (1.0 - (u ** -1)), 0.0)

        return sigma

    def spin_forbidden(self, eps_mn, alpha_mn, equation_type, g_n, g_m):
        """
        Formula for Spin-Forbidden Transitions (S).
        Based on Vlcek (1989) Eqs (6), (7), (8).

        Parameters:
            eps_mn (float): Threshold energy in eV.
            alpha_mn (float): Fitting parameter.
            equation_type (int): 0, 1, or 2 depending on the specific formula used.
            g_n (float): Statistical weight of the lower level n.
            g_m (float): Statistical weight of the upper level m.

        Returns:
            callable: A function sigma(E) in cm^2.
        """
        if equation_type == 0:
            # Equation type 0: Used for certain forbidden transitions
            prefactor = 5.83e-15 * (g_m / g_n) * alpha_mn

            def sigma(E):
                if np.isscalar(E):
                    if E <= eps_mn: return 0.0
                    u = E / eps_mn
                    return prefactor * ((1.0 - (1.0 / u)) ** 0.46) * ((eps_mn * u) ** -0.54)
                
                E_arr = np.asarray(E)
                u = np.where(E_arr > eps_mn, E_arr / eps_mn, 1.0)
                return np.where(E_arr > eps_mn, prefactor * ((1.0 - (1.0 / u)) ** 0.46) * ((eps_mn * u) ** -0.54), 0.0)

        elif equation_type == 1:
            # Equation type 1: Used for other forbidden transitions
            prefactor = 8.13e-16 * (g_m / g_n) * alpha_mn

            def sigma(E):
                if np.isscalar(E):
                    if E <= eps_mn: return 0.0
                    u = E / eps_mn
                    # 1e-10 prevents numerical issues when u is close to 1.0
                    term1 = (max(1.0 - (1.0 / u), 1e-10)) ** -0.04
                    term2 = (eps_mn * u) ** -1.04
                    return prefactor * term1 * term2
                E_arr = np.asarray(E)
                u = np.where(E_arr > eps_mn, E_arr / eps_mn, 1.0)
                term1 = np.where(u > 1.0, (1.0 - (1.0 / u)), 1e-10) ** -0.04
                return np.where(E_arr > eps_mn, prefactor * term1 * ((eps_mn * u) ** -1.04), 0.0)

        else:
            # Equation type 2: Generic spin-forbidden formula
            prefactor = self.four_pi_a02 * alpha_mn

            def sigma(E):
                if np.isscalar(E):
                    if E <= eps_mn: return 0.0
                    u = E / eps_mn
                    return prefactor * (u ** -3) * (1.0 - (u ** -2))
                E_arr = np.asarray(E)
                u = np.where(E_arr > eps_mn, E_arr / eps_mn, 1.0)
                return np.where(E_arr > eps_mn, prefactor * (u ** -3) * (1.0 - (u ** -2)), 0.0)

        return sigma

    # =========================================================================
    # IONIZATION & RECOMBINATION
    # =========================================================================

    def e_ionization(self, eps_ni, xi_n, alpha_n, beta_n):
        """
        Electron impact ionization cross section.
        Based on Vlcek (1989) Eq (13).

        Parameters:
            eps_ni (float): Ionization potential of level n in eV.
            xi_n (float): Number of equivalent electrons in the outer shell.
            alpha_n (float): Fitting parameter.
            beta_n (float): Fitting parameter.

        Returns:
            callable: A function sigma(E) in cm^2.
        """
        prefactor = self.four_pi_a02 * ((E_ION_H / eps_ni) ** 2) * xi_n * alpha_n

        def sigma(E):
            if np.isscalar(E):
                if E <= eps_ni: return 0.0
                u = E / eps_ni
                return prefactor * (u ** -2) * (u - 1.0) * np.log(1.25 * beta_n * u)
            else:
                E_arr = np.asarray(E)
                u = np.where(E_arr > eps_ni, E_arr / eps_ni, 1.0)
                return np.where(E_arr > eps_ni, prefactor * (u ** -2) * (u - 1.0) * np.log(1.25 * beta_n * u), 0.0)

        return sigma

    def photoionization(self, n, eps_n, gamma_n=1.0, eps_4s_mean=None):
        """
        Photoionization cross section.
        Used for radiative recombination calculations via detailed balance.

        Parameters:
            n (int): Level ID.
            eps_n (float): Ionization potential of level n in eV.
            gamma_n (float): Correction factor (default 1.0).
            eps_4s_mean (float, optional): Mean threshold for 4s levels.

        Returns:
            callable: A function sigma(h_nu) in cm^2.
        """
        if n == 1:
            # Ground state photoionization
            def sigma(h_nu):
                if np.isscalar(h_nu):
                    if h_nu < eps_n: return 0.0
                    return 3.5e-17 if h_nu <= 2 * E_ION_H else 2.8e-16 * (E_ION_H / h_nu) ** 3
                h_arr = np.asarray(h_nu)
                return np.where(h_arr < eps_n, 0.0,
                                np.where(h_arr <= 2 * E_ION_H, 3.5e-17, 2.8e-16 * (E_ION_H / h_arr) ** 3))

        elif 2 <= n <= 5:
            # Excited 4s states photoionization
            if eps_4s_mean is None: raise ValueError("eps_4s_mean must be provided for 4s levels")
            limit = 0.59 * E_ION_H

            def sigma(h_nu):
                if np.isscalar(h_nu):
                    if h_nu < eps_4s_mean: return 0.0
                    return 2.8e-18 * gamma_n if h_nu <= limit else 7.91e-18 * gamma_n * (
                                (eps_4s_mean / E_ION_H) ** 2.5) * ((E_ION_H / h_nu) ** 3)
                h_arr = np.asarray(h_nu)
                return np.where(h_arr < eps_4s_mean, 0.0,
                       np.where(h_arr <= limit, 2.8e-18 * gamma_n,
                                7.91e-18 * gamma_n * ((eps_4s_mean / E_ION_H) ** 2.5) * ((E_ION_H / h_arr) ** 3))
                                )

        else:
            # Other excited states photoionization
            def sigma(h_nu):
                if np.isscalar(h_nu):
                    if h_nu < eps_n: return 0.0
                    return 7.91e-18 * gamma_n * ((eps_n / E_ION_H) ** 2.5) * ((E_ION_H / h_nu) ** 3)
                h_arr = np.asarray(h_nu)
                return np.where(h_arr < eps_n, 0.0,
                                7.91e-18 * gamma_n * ((eps_n / E_ION_H) ** 2.5) * ((E_ION_H / h_arr) ** 3))

        return sigma

    # =========================================================================
    # DISPATCHER
    # =========================================================================

    def evaluate_e_exc(self, trans_type, eps_mn, alpha_mn, beta_mn=None, equation_type=0, g_n=1.0, g_m=1.0):
        """
        Master routing function that returns the appropriate sigma(E) function 
        based on the transition type.

        Parameters:
            trans_type (str): 'A' (Allowed), 'P' (Parity-forbidden), 'S' (Spin-forbidden).
            eps_mn (float): Threshold energy in eV.
            alpha_mn (float): Fitting parameter.
            beta_mn (float, optional): Fitting parameter for 'A' type.
            equation_type (int): Sub-type for 'S' transitions.
            g_n (float): Statistical weight of lower level.
            g_m (float): Statistical weight of upper level.

        Returns:
            callable: The selected sigma(E) function.
        """
        if trans_type == 'A':
            return self.allowed(eps_mn, alpha_mn, beta_mn)
        elif trans_type == 'P':
            return self.parity_forbidden(eps_mn, alpha_mn)
        elif trans_type == 'S':
            return self.spin_forbidden(eps_mn, alpha_mn, equation_type, g_n, g_m)
        else:
            # Fallback for unknown types
            return lambda E: 0.0