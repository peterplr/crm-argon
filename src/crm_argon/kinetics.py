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
from scipy.integrate import quad
from scipy.special import erf

from .constants import (
    m_e, e, h, k_B, c, M_AR, AMU_TO_KG, J_TO_EV, EV_TO_J
)


class Kinetics:
    """
    A mathematical integrator for Collisional-Radiative rate coefficients.
    Executes equations strictly in SI units to prevent dimensional errors
    and returns final rate coefficients in standard plasma units (cm^3/s or cm^6/s).
    """

    def __init__(self):
        # Reduced mass for Argon-Argon collisions (m_12 = m_Ar / 2)
        self.m_12_gas = (M_AR * AMU_TO_KG) / 2.0  # [kg]
        self.k_B_eV = J_TO_EV * k_B

    @staticmethod
    def _calc_prefactor(Te_eV):
        """
        Calculates the 8 * pi * (k * Te / m_e)^2 prefactor.
        Units (SI): [m^4 / s^4]
        """
        Te_joules = Te_eV * e
        return 8.0 * np.pi * (Te_joules / m_e) ** 2

    # =========================================================================
    # ELECTRON COLLISION RATES
    # =========================================================================

    def get_exc_rate_coeff(self, sigma_func, eedf_func, eps_mn_eV, Te_eV):
        """
        Calculates the electron impact excitation rate coefficient C_nm for the process m -> n.
        
        This method computes the rate coefficient by integrating the product of the cross section,
        the energy-weighted EEDF, and the electron velocity over the energy range starting
        from the excitation threshold.

        Parameters:
            sigma_func (callable): Function sigma(E) returning the cross section in cm^2.
            eedf_func (callable): Function f(u) representing the Electron Energy Distribution Function,
                                  where u = E / Te.
            eps_mn_eV (float): Threshold energy for the transition m -> n in eV.
            Te_eV (float): Electron temperature in eV.

        Returns:
            float: The rate coefficient C_nm in units of cm^3/s.
        """
        u_mn = eps_mn_eV / Te_eV
        prefactor = self._calc_prefactor(Te_eV)

        def integrand(u):
            # Convert cross section from cm^2 to m^2
            sigma_m2 = sigma_func(u * Te_eV) * 1e-4
            return eedf_func(u) * sigma_m2 * u

        # Integrate from threshold u_mn to infinity
        integral, _ = quad(integrand, u_mn, np.inf, limit=100)

        return prefactor * integral * 1e6

    def get_de_exc_rate_coeff(self, sigma_func, eedf_func, eps_mn_eV, Te_eV, g_m, g_n):
        """
        Calculates the electron impact de-excitation rate coefficient F_mn for the process n -> m (n > m).
        
        Uses the principle of detailed balancing (Klein-Rosseland relation) to derive
        the de-excitation rate from the excitation cross section.

        Parameters:
            sigma_func (callable): Function sigma(E) returning the excitation cross section m -> n in cm^2.
            eedf_func (callable): Function f(u) representing the EEDF.
            eps_mn_eV (float): Threshold energy for the excitation m -> n in eV.
            Te_eV (float): Electron temperature in eV.
            g_m (float): Statistical weight of the lower level m.
            g_n (float): Statistical weight of the upper level n.

        Returns:
            float: The de-excitation rate coefficient F_mn in units of cm^3/s.
        """
        u_nm = eps_mn_eV / Te_eV
        prefactor = self._calc_prefactor(Te_eV)
        weight_ratio = g_m / g_n

        def integrand(u):
            # Convert cross section from cm^2 to m^2
            sigma_m2 = sigma_func(u * Te_eV) * 1e-4
            # For de-excitation, the EEDF is evaluated at the energy shifted by the threshold
            return eedf_func(u - u_nm) * sigma_m2 * u

        # Integrate from threshold u_nm to infinity
        integral, _ = quad(integrand, u_nm, np.inf, limit=100)

        return prefactor * weight_ratio * integral * 1e6

    def get_ioniz_coeff(self, sigma_func, eedf_func, eps_ion_eV, Te_eV):
        """
        Calculates the electron impact ionization rate coefficient S_n for the process n -> ion.

        Parameters:
            sigma_func (callable): Function sigma(E) returning the ionization cross section in cm^2.
            eedf_func (callable): Function f(u) representing the EEDF.
            eps_ion_eV (float): Ionization potential of level n in eV.
            Te_eV (float): Electron temperature in eV.

        Returns:
            float: The ionization rate coefficient S_n in units of cm^3/s.
        """
        u_n = eps_ion_eV / Te_eV
        prefactor = self._calc_prefactor(Te_eV)

        def integrand(u):
            # Convert cross section from cm^2 to m^2
            sigma_m2 = sigma_func(u * Te_eV) * 1e-4
            return eedf_func(u) * sigma_m2 * u

        # Integrate from threshold u_n to infinity
        integral, _ = quad(integrand, u_n, np.inf, limit=100)

        return prefactor * integral * 1e6

    def get_recomb_coeff(self, sigma_func, eedf_func, eps_ion_eV, Te_eV, g_n, g_ion):
        """
        Calculates the three-body recombination rate coefficient O_n for the process ion + e + e -> n + e.
        
        Derived from the ionization cross section using the principle of detailed balancing.

        Parameters:
            sigma_func (callable): Function sigma(E) returning the ionization cross section in cm^2.
            eedf_func (callable): Function f(u) representing the EEDF.
            eps_ion_eV (float): Ionization potential of level n in eV.
            Te_eV (float): Electron temperature in eV.
            g_n (float): Statistical weight of the neutral level n.
            g_ion (float): Statistical weight of the ion ground state.

        Returns:
            float: The three-body recombination rate coefficient O_n in units of cm^6/s.
        """
        u_n = eps_ion_eV / Te_eV
        prefactor = self._calc_prefactor(Te_eV)
        weight_ratio = g_n / (2.0 * g_ion)

        Te_joules = Te_eV * e
        # Thermal de Broglie wavelength cubed
        de_broglie_m3 = ((h ** 2) / (2.0 * np.pi * m_e * Te_joules)) ** 1.5

        def integrand(u):
            # Convert cross section from cm^2 to m^2
            sigma_m2 = sigma_func(u * Te_eV) * 1e-4
            return eedf_func(u - u_n) * sigma_m2 * u

        # Integrate from threshold u_n to infinity
        integral, _ = quad(integrand, u_n, np.inf, limit=100)

        return prefactor * weight_ratio * de_broglie_m3 * integral * 1e12

    # ==========================================
    # PHOTORECOMBINATION
    # ==========================================

    def get_photorecomb_coeff(self, sigma_func, eedf_func, eps_m_eV, Te_eV, g_m, g_ion):
        """
        Calculates the radiative (photo) recombination rate coefficient R_m for the process ion + e -> m + h*nu.

        Parameters:
            sigma_func (callable): Function sigma(E) returning the photoionization cross section for level m.
            eedf_func (callable): Function f(u) representing the EEDF.
            eps_m_eV (float): Ionization potential of level m in eV.
            Te_eV (float): Electron temperature in eV.
            g_m (float): Statistical weight of level m.
            g_ion (float): Statistical weight of the ion ground state.

        Returns:
            float: The photorecombination rate coefficient R_m in units of cm^3/s.
        """

        # The exact derivation: e^2 from E_ph^2 conversion, (e * Te_eV) from dE_e conversion
        energy_scalars = (e ** 3) * Te_eV
        weight_ratio = g_m / (2.0 * g_ion)
        prefactor_SI = (8.0 * np.pi * energy_scalars * weight_ratio) / (m_e ** 3 * c ** 2)

        def integrand(u):
            # Evaluate photon energy natively in eV for the cross-section function
            E_ph_eV = (u * Te_eV) + eps_m_eV
            # Convert cross section from cm^2 to m^2
            sigma_m2 = sigma_func(E_ph_eV) * 1e-4
            return eedf_func(u) * sigma_m2 * (E_ph_eV ** 2)

        integral_SI, _ = quad(integrand, 0.0, np.inf, limit=100)

        return prefactor_SI * integral_SI * 1e6

    # =========================================================================
    # ATOM-ATOM HEAVY PARTICLE COLLISIONS
    # =========================================================================

    def _get_b(self, eps_mn_eV, c):
        """
        Returns the b_mn parameter used in Drawin's formula for heavy particle collisions.
        
        Units: [cm^2 / eV]
        """
        return c * (eps_mn_eV ** -2.26)

    def get_atom_exc_coeff(self, eps_mn_eV, Tg_K, c):
        """
        Calculates the atom-atom impact excitation rate coefficient K_nm for the process m -> n.
        Based on Drawin's formula for heavy particle impact.

        Parameters:
            eps_mn_eV (float): Threshold energy for the transition m -> n in eV.
            Tg_K (float): Gas temperature in K.
            c (float): Fitting constant for the specific transition.

        Returns:
            float: The rate coefficient K_nm in units of cm^3/s.
        """
        b_mn_cm2_eV = self._get_b(eps_mn_eV, c)

        # Thermal velocity of the reduced mass system
        v_th_m_s = np.sqrt((2.0 * k_B * Tg_K) / (np.pi * self.m_12_gas))
        # Convert from m/s to cm/s
        v_th_cm_s = v_th_m_s * 100.0

        # Mean energy term in eV
        energy_term_eV = eps_mn_eV + (2.0 * self.k_B_eV * Tg_K)
        # Boltzmann factor for the threshold
        exp_term = np.exp(-eps_mn_eV / (self.k_B_eV * Tg_K))

        return 2.0 * v_th_cm_s * b_mn_cm2_eV * energy_term_eV * exp_term

    def get_atom_de_exc_coeff(self, K_nm, eps_mn_eV, Tg_K, g_m, g_n):
        """
        Calculates the atom-atom de-excitation rate coefficient L_mn for the process n -> m (n > m).
        Derived from the excitation rate coefficient using the principle of detailed balancing.

        Parameters:
            K_nm (float): Excitation rate coefficient m -> n in cm^3/s.
            eps_mn_eV (float): Threshold energy for the excitation m -> n in eV.
            Tg_K (float): Gas temperature in K.
            g_m (float): Statistical weight of level m.
            g_n (float): Statistical weight of level n.

        Returns:
            float: The de-excitation rate coefficient L_mn in units of cm^3/s.
        """
        exp_term = np.exp(eps_mn_eV / (self.k_B_eV * Tg_K))
        return K_nm * (g_m / g_n) * exp_term

    def get_atom_ioniz_coeff(self, eps_ni_eV, Tg_K, c):
        """
        Calculates the atom impact ionization rate coefficient V_n for the process n -> ion.
        Uses the same Drawin-based logic as excitation.

        Parameters:
            eps_ni_eV (float): Ionization potential of level n in eV.
            Tg_K (float): Gas temperature in K.
            c (float): Fitting constant.

        Returns:
            float: The ionization rate coefficient V_n in units of cm^3/s.
        """
        return self.get_atom_exc_coeff(eps_ni_eV, Tg_K, c)

    def get_atom_recomb_coeff(self, V_n, eps_ni_eV, Tg_K, Te_eV, g_n, g_ion):
        """
        Calculates the atom impact three-body recombination rate coefficient W_n 
        for the process ion + Ar + e -> n + Ar + e.

        Parameters:
            V_n (float): Ionization rate coefficient n -> ion in cm^3/s.
            eps_ni_eV (float): Ionization potential of level n in eV.
            Tg_K (float): Gas temperature in K.
            Te_eV (float): Electron temperature in eV.
            g_n (float): Statistical weight of level n.
            g_ion (float): Statistical weight of the ion ground state.

        Returns:
            float: The recombination rate coefficient W_n in units of cm^6/s.
        """
        Te_joules = Te_eV * EV_TO_J

        # Thermal de Broglie wavelength cubed (standard volume for detailed balance)
        de_broglie_m3 = ((h ** 2) / (2.0 * np.pi * m_e * Te_joules)) ** 1.5
        de_broglie_cm3 = de_broglie_m3 * 1e6

        exp_term = np.exp(eps_ni_eV / (self.k_B_eV * Tg_K))

        return V_n * (g_n / (2.0 * g_ion)) * de_broglie_cm3 * exp_term

    # =========================================================================
    # RADIATION TRAPPING (ESCAPE FACTORS)
    # =========================================================================

    def get_escape_factor(self, E_mn_eV, g_m, A_mn, n_gas_cm3, R_cm, Tg_K):
        """
        Calculates the escape factor Lambda_mn for transitions from level n to level m.
        Currently implemented for transitions to the ground state (m=1).
        Uses the Bogaerts (1998) cylindrical tube formulation with combined Doppler and
        collisional broadening.

        Parameters:
            E_mn_eV (float): Transition energy between levels n and m in eV.
            g_m (float): Statistical weight of the lower level m.
            A_mn (float): Einstein A coefficient for the transition n -> m in s^-1.
            n_gas_cm3 (float): Density of the lower level m (usually ground state) in cm^-3.
            R_cm (float): Radius of the cylindrical discharge tube in cm.
            Tg_K (float): Gas temperature in K.

        Returns:
            float: Escape factor Lambda_mn (ranging from 0.0 to 1.0).
        """
        # Optical depth at the line center (k0 * R)
        # Based on Vlcek (1989) and Bogaerts (1998)
        k0_R_prefactor = (2.1e-17 * g_m) / ((E_mn_eV ** 3) * np.sqrt(Tg_K))
        k0_R = k0_R_prefactor * A_mn * n_gas_cm3 * R_cm

        # Bogaerts formula is an asymptotic approximation for optically thick lines.
        # If optical depth is very low (k0_R <= 1.0), the line is completely optically
        # thin, and the escape factor is 1.0.
        if k0_R <= 1.01:
            return 1.0

        # Damping coefficient (a) representing the ratio of Lorentzian to Doppler broadening
        bracket_term = 1.0 + ((3.225e-14 / (E_mn_eV ** 3)) * g_m * n_gas_cm3)
        fraction = 4.839e-9 / (E_mn_eV * np.sqrt(Tg_K))
        a = A_mn * bracket_term * fraction

        # Transmission coefficients for different broadening regimes
        ln_k0R = np.log(k0_R)

        T_D = 1.0 / (k0_R * np.sqrt(np.pi * ln_k0R))
        T_C = np.sqrt(a / (np.sqrt(np.pi) * k0_R))
        T_CD = (2.0 * a) / (np.pi * np.sqrt(ln_k0R))

        # Calculating the combined Escape Factor (Lambda)
        exp_arg = -(np.pi * T_CD ** 2) / (4.0 * T_C ** 2)
        erf_arg = (np.sqrt(np.pi) * T_CD) / (2.0 * T_C)

        Lambda = (1.9 * T_D * np.exp(exp_arg)) + (1.3 * T_C * erf(erf_arg))

        # Physical limit: Escape factor cannot exceed 1.0 (optically thin limit)
        return min(Lambda, 1.0)