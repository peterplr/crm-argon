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

import sys
# Use standard library tomllib for Python 3.11+. Fallback for older versions.
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import numpy as np

from .constants import m_e, e, h


class PhysicsEngine:
    """
    The central physics engine of the Collisional-Radiative model for Argon.

    This class is responsible for assembling the rate matrix and solving for the steady-state
    populations of excited states. It evaluates dynamic transition conditions from the
    configuration file and utilizes the database, cross sections, and kinetics modules
    to calculate individual transition rates.
    """

    def __init__(self, config_dict, database, cross_sections, rate_calculator, steady_state_solver):
        """
        Initializes an instance of the class with configuration, database, cross sections, rate
        calculator, and steady state solver. Sets up matrix indexing and initializes relevant
        attributes for subsequent calculations.

        Args:
            config_dict (dict): Configuration dictionary containing model and process
                parameters.
            database: Database object storing relevant data for the model.
            cross_sections: Object providing cross-sectional data for the processes.
            rate_calculator: Calculator object for computing rates in the model.
            steady_state_solver: Solver object for determining steady state solutions.

        Attributes:
            config (dict): Stores the configuration dictionary for the model setup.
            procs (dict): Dictionary of processes defined in the configuration.
            active_levels (list): List of level indices actively considered in the model.
            num_levels (int): Number of active levels in the model.
            idx_map (dict): Mapping of level indices to internal representation for matrix
                operations.
            db: Reference to the database object.
            xs: Reference to the cross-section data object.
            calc: Reference to the rate calculator object.
            ss_solver: Reference to the steady state solver object.
        """
        self.config = config_dict
        self.procs = self.config.get('processes', {})

        # Setup matrix indexing
        max_lvl = self.config.get('model', {}).get('maximum_level', 65)
        self.active_levels = list(range(2, max_lvl + 1))
        self.num_levels = len(self.active_levels)
        self.idx_map = {level: i for i, level in enumerate(self.active_levels)}

        self.db = database
        self.xs = cross_sections
        self.calc = rate_calculator
        self.solver = steady_state_solver

    # ==========================================
    # CONDITION EVALUATION LOGIC
    # ==========================================

    def _is_intercombination(self, m, n):
        """
        Determines if a transition between two levels is an intercombination transition.

        An intercombination transition occurs between states with different ion core
        angular momentum (j_c = 1/2 vs j_c = 3/2).

        Args:
            m (int): Level ID of the first state.
            n (int): Level ID of the second state.

        Returns:
            bool: True if the transition is intercombination, False otherwise.
        """
        core_m = self.db.query('levels', lower_level=m).get('qn_core', [])
        core_n = self.db.query('levels', lower_level=n).get('qn_core', [])

        if not core_m or not core_n:
            return False

        return set(core_m).isdisjoint(set(core_n))

    def _check_conditions(self, process, sub_process=None, **kwargs):
        """
        Evaluates dynamic logic conditions from the TOML configuration to decide if a
        transition should be included in the model.

        Args:
            process (str): The main process name (e.g., 'electron_excitation').
            sub_process (str, optional): The specific sub-process category.
            **kwargs: Additional parameters like 'lower_level' and 'upper_level'.

        Returns:
            bool: True if the transition satisfies all conditions, False otherwise.
        """
        # Global Process Check: Is this physics mechanism turned on at all?
        if not self.procs.get(process, True):
            return False

        # Extract m and n safely from the kwargs passed by the physics engine
        m = kwargs.get('lower_level')
        n = kwargs.get('upper_level')

        # Sub-Process Boundary Check
        if sub_process and m is not None and n is not None:
            # Navigate safely through the nested TOML dictionaries
            conditions = self.config.get('transition_conditions', {}).get(process, {}).get(sub_process, {})

            if not conditions:
                # If no specific limits are defined for this sub_process, assume it is fully allowed
                return True

            # Fast, explicit boundary checks
            if 'm_max' in conditions and m > conditions['m_max']: return False
            if 'm_min' in conditions and m < conditions['m_min']: return False
            if 'n_max' in conditions and n > conditions['n_max']: return False
            if 'n_min' in conditions and n < conditions['n_min']: return False

            # Fast explicit list checks
            if 'm_list' in conditions and m not in conditions['m_list']: return False
            if 'n_list' in conditions and n not in conditions['n_list']: return False

        return True


    # =========================================================================
    # RATE HELPERS
    # =========================================================================

    def _calc_electron_excitation(self, m, n, Te_eV, eedf_func):
        """
        Calculates electron impact transition rates between two levels.
        Enforces detailed balance: if excitation is calculated, de-excitation is too.
        If the pathway is turned off or blocked by conditions, both return 0.0.

        Parameters:
            m (int): Lower energy level ID.
            n (int): Upper energy level ID.
            Te_eV (float): Electron temperature in eV.
            eedf_func (callable): EEDF function.

        Returns:
            tuple: (C_nm, C_mn) in cm^3/s.
                   C_nm = Excitation (m -> n)
                   C_mn = De-excitation (n -> m)
        """
        # EARLY RETURN: Global toggle
        if not self.procs.get('electron_excitation', True):
            return 0.0, 0.0

        # Satisfy database integer constraint while preserving m/n energy logic
        idx_low, idx_high = min(m, n), max(m, n)
        params_list = self.db.query('excitation', lower_level=idx_low, upper_level=idx_high)

        total_C_nm, total_C_mn = 0.0, 0.0

        if not params_list:
            return total_C_nm, total_C_mn

        type_map = {'A': 'allowed', 'P': 'parity_forbidden', 'S': 'spin_forbidden'}
        is_inter = self._is_intercombination(m, n)

        lvl_m = self.db.query('levels', lower_level=m)
        lvl_n = self.db.query('levels', lower_level=n)
        g_m = lvl_m['g']
        g_n = lvl_n['g']

        # Ground state special case: statistical weight depends on ion core
        if m == 1:
            core_n_list = lvl_n.get('qn_core', [])
            dominant_core = core_n_list[0] if core_n_list else None
            g_m = 1.0 / 3.0 if dominant_core == '1/2' else 2.0 / 3.0

        for params in params_list:
            trans_type = params.get('type')
            if not trans_type: continue

            sub_cat = type_map.get(trans_type)

            # EXPLICIT COUPLING: Evaluate TOML conditions. If blocked, skip both forward and reverse.
            if not self._check_conditions('electron_excitation', sub_process=sub_cat,
                                          lower_level=idx_low, upper_level=idx_high):
                continue

            eps_threshold = params['excitation_energy']

            # Retrieve the appropriate cross-section function
            sigma_func = self.xs.evaluate_e_exc(
                trans_type=trans_type,
                eps_mn=eps_threshold,
                alpha_mn=params['alpha'],
                beta_mn=params.get('beta', 0.0),
                equation_type=params.get('equation', 0),
                g_n=g_m,
                g_m=g_n
            )

            # Integrate for rate coefficients
            C_nm = self.calc.get_exc_rate_coeff(sigma_func, eedf_func, eps_threshold, Te_eV)
            C_mn = self.calc.get_de_exc_rate_coeff(sigma_func, eedf_func, eps_threshold, Te_eV, g_m, g_n)

            total_C_nm += C_nm  # m -> n
            total_C_mn += C_mn  # n -> m

        return total_C_nm, total_C_mn

    def _calc_atom_excitation(self, m, n, Tg_K):
        """
        Calculates atom-atom impact transition rates between two levels.
        Enforces detailed balance identically to electron excitation.

        Parameters:
            m (int): Lower energy level ID.
            n (int): Upper energy level ID.
            Tg_K (float): Gas temperature in K.

        Returns:
            tuple: (K_nm, K_mn) in cm^3/s.
                   K_nm = Excitation (m -> n)
                   K_mn = De-excitation (n -> m)
        """
        # EARLY RETURN & COUPLING: Check global toggle and specific TOML conditions
        if not self.procs.get('atom_excitation', True) or \
                not self._check_conditions('atom_excitation', sub_process='general', lower_level=m, upper_level=n):
            return 0.0, 0.0

        E_m = self.db.query('levels', lower_level=m)['excitation_energy']
        E_n = self.db.query('levels', lower_level=n)['excitation_energy']
        eps_threshold = abs(E_n - E_m)

        if eps_threshold < 1e-4:
            return 0.0, 0.0

        c_const = self._get_atom_c_constant(m, n)
        lvl_m = self.db.query('levels', lower_level=m)
        lvl_n = self.db.query('levels', lower_level=n)

        g_m_true = lvl_m['g']
        g_n = lvl_n['g']

        # Calculate standard excitation and detailed balance using true statistical weight
        K_nm = self.calc.get_atom_exc_coeff(eps_threshold, Tg_K, c_const)
        K_mn = self.calc.get_atom_de_exc_coeff(K_nm, eps_threshold, Tg_K, g_m_true, g_n)

        # Apply specific corrections if the lower state is the ground state
        if m == 1:
            K_nm *= 0.5
            K_mn *= 0.5  # Scale both sides to mathematically maintain detailed balance

        return K_nm, K_mn

    def _get_atom_c_constant(self, m, n):
        """
        Determines the fitting constant 'c' for atom-atom collision cross sections
        based on the transition type (intercombination vs. intra-core).
        Follows the empirical values suggested in Vlcek (1989).

        Args:
            m (int): Level ID of the first state.
            n (int): Level ID of the second state.

        Returns:
            float: The fitting constant 'c' in cm^2/eV.
        """
        if m in [2, 3, 4, 5] and n in [2, 3, 4, 5]:
            if self._is_intercombination(m, n):
                return 4.80e-22  # Intercombination between 4s levels
            else:
                return 1.79e-20  # Transitions within the same core system
        return 8.69e-18  # Default constant for other transitions

    def _calc_radiative_rates(self, n, m, n_1_cm3, R_cm, Tg_K):
        """
        Retrieves the spontaneous emission rate (Einstein A) and calculates the
        escape factor for radiation trapping.

        Parameters:
            n (int): Upper energy level ID.
            m (int): Lower energy level ID.
            n_1_cm3 (float): Ground state density in cm^-3.
            R_cm (float): Plasma radius in cm.
            Tg_K (float): Gas temperature in K.

        Returns:
            tuple: (A_mn, Lambda_mn) in s^-1 and dimensionless (Transition n -> m).
        """
        A_mn, Lambda_mn = 0.0, 1.0

        # Check global toggle AND level-specific TOML conditions
        if not self.procs.get('spontaneous_emission', True) or \
                not self._check_conditions('spontaneous_emission', sub_process='general', lower_level=n,
                                           upper_level=m):
            return A_mn, Lambda_mn

        # Access pre-indexed emission matrix
        A_mn = self.db.emission_matrix[n - 1, m - 1]

        if A_mn == 0.0:
            return A_mn, Lambda_mn

        # Apply radiation trapping only for transitions to the ground state (m=1)
        if m == 1 and self.procs.get('radiation_trapping', True):
            lvl_n = self.db.query('levels', lower_level=n)
            Lambda_mn = self.calc.get_escape_factor(lvl_n['excitation_energy'], lvl_n['g'], A_mn, n_1_cm3, R_cm,
                                                    Tg_K)

        return A_mn, Lambda_mn

    def _calc_ionization_rates(self, n, Te_eV, Tg_K, eedf_func):
        """
        Calculates all ionization and recombination rates associated with level n.
        Explicitly couples recombination to ionization: 3-body recombination is only
        permitted if the corresponding forward ionization pathway is active.

        Parameters:
            n (int): Level ID.
            Te_eV (float): Electron temperature in eV.
            Tg_K (float): Gas temperature in K.
            eedf_func (callable): EEDF function.

        Returns:
            tuple: (S_n, O_n, V_n, W_n) - Electron ionization, 3-body e-recomb,
                   Atom ionization, 3-body atom-recomb.
        """
        S_n, O_n, V_n, W_n = 0.0, 0.0, 0.0, 0.0

        # Evaluate forward ionization conditions
        allow_e_ion = self.procs.get('electron_ionization', True) and \
                      self._check_conditions('electron_ionization', sub_process='general', lower_level=n)

        allow_a_ion = self.procs.get('atom_ionization', True) and \
                      self._check_conditions('atom_ionization', sub_process='general', lower_level=n)

        # Evaluate reverse recombination conditions (dependent on forward process)
        allow_3b_e_rec = allow_e_ion and self.procs.get('three_body_recombination', True)
        allow_3b_a_rec = allow_a_ion and self.procs.get('atom_recombination', True)

        # EARLY RETURN: If all pathways are blocked, skip DB queries entirely
        if not any([allow_e_ion, allow_a_ion, allow_3b_e_rec, allow_3b_a_rec]):
            return S_n, O_n, V_n, W_n

        ion_params = self.db.query('ionization', lower_level=n)
        if ion_params and ion_params.get('alpha_ion'):
            eps_ion = ion_params['ionization_energy']
            g_n = self.db.query('levels', lower_level=n)['g']
            core_n_list = self.db.query('levels', lower_level=n).get('qn_core', [])
            dominant_core = core_n_list[0] if core_n_list else None
            g_plus = 2.0 if dominant_core == '1/2' else 4.0

            # 1. Electron Impact Ionization
            if allow_e_ion:
                sigma_ion = self.xs.e_ionization(
                    eps_ni=eps_ion,
                    xi_n=ion_params['xi_ion'],
                    alpha_n=ion_params['alpha_ion'],
                    beta_n=ion_params['beta_ion']
                )
                S_n = self.calc.get_ioniz_coeff(sigma_ion, eedf_func, eps_ion, Te_eV)

                # 2. Three-Body Electron Recombination (Detailed Balance)
                if allow_3b_e_rec:
                    O_n = self.calc.get_recomb_coeff(sigma_ion, eedf_func, eps_ion, Te_eV, g_n, g_plus)

            # 3. Atom Impact Ionization
            if allow_a_ion:
                V_n = self.calc.get_atom_ioniz_coeff(eps_ion, Tg_K, 8.69e-18)

                if n == 1:
                    V_n *= 0.5  # Correct for Ar-Ar collisions

                # 4. Atom Impact Three-Body Recombination
                if allow_3b_a_rec:
                    W_n = self.calc.get_atom_recomb_coeff(V_n, eps_ion, Tg_K, Te_eV, g_n, g_plus)

        return S_n, O_n, V_n, W_n

    def _calc_photorecombination(self, n, Te_eV, eedf_func, eps_4s_mean=None):
        """
        Calculates the radiative recombination rate for level n.

        Parameters:
            n (int): Target Level ID.
            Te_eV (float): Electron temperature in eV.
            eedf_func (callable): EEDF function.
            eps_4s_mean (float, optional): Mean 4s threshold for specific cross-section formulas.

        Returns:
            float: Photorecombination rate R_n in cm^3/s.
        """
        # EARLY RETURN: Global toggle & TOML condition
        if not self.procs.get('photorecombination', True) or \
                not self._check_conditions('photorecombination', sub_process='general', lower_level=n):
            return 0.0

        photo_params = self.db.query('photoionization', lower_level=n)
        ion_params = self.db.query('ionization', lower_level=n)

        if photo_params and ion_params:
            eps_ion = ion_params['ionization_energy']
            gamma_n = photo_params['gamma_p']
            g_n = self.db.query('levels', lower_level=n)['g']
            core_n_list = self.db.query('levels', lower_level=n).get('qn_core', [])
            dominant_core = core_n_list[0] if core_n_list else None
            g_plus = 2.0 if dominant_core == '1/2' else 4.0

            sigma_photo = self.xs.photoionization(
                n=n,
                eps_n=eps_ion,
                gamma_n=gamma_n,
                eps_4s_mean=eps_4s_mean
            )
            return self.calc.get_photorecomb_coeff(sigma_photo, eedf_func, eps_ion, Te_eV, g_n, g_plus)

        return 0.0

    def _calc_quenching_rates(self, n, Tg_K, n_1_cm3, R_cm):
        """
        Calculates metastable diffusion and three-body quenching losses.
        These are strictly treated as loss terms from the given level.

        Parameters:
            n (int): Energy level ID.
            Tg_K (float): Gas temperature in K.
            n_1_cm3 (float): Ground state density in cm^-3.
            R_cm (float): Plasma radius in cm.

        Returns:
            tuple: (loss_diff, loss_3b) in s^-1.
        """
        loss_diff, loss_3b = 0.0, 0.0

        mq_data = self.db.query('metastable_quenching', lower_level=n)
        if not mq_data:
            return loss_diff, loss_3b

        d_n = mq_data.get('d_n', 0.0)
        B_n = mq_data.get('B_n', 0.0)

        # Only evaluate if the global toggle is True AND the database has non-zero parameters
        if self.procs.get('diffusion_quenching', True) and d_n > 0:
            loss_diff = d_n * (Tg_K ** 0.73) / (n_1_cm3 * (R_cm ** 2))

        if self.procs.get('three_body_quenching', True) and B_n > 0:
            loss_3b = B_n * (n_1_cm3 ** 2)

        return loss_diff, loss_3b


    # ==========================================
    # MATRIX ASSEMBLY (VLCEK EQ 1-5)
    # ==========================================

    def build_and_solve(self, Te_eV, Tg_K, n_e, n_1, n_ion, R_cm, eedf_func):
        """
        Main method to assemble the CR rate matrix and solve for excited state populations.
        Follows the logic of Vlcek (1989) Equation (1).

        Parameters:
            Te_eV (float): Electron temperature in eV.
            Tg_K (float): Gas temperature in K.
            n_e (float): Electron density in cm^-3.
            n_1 (float): Ground state density in cm^-3.
            n_ion (float): Ion density in cm^-3 (usually equal to n_e).
            R_cm (float): Plasma radius in cm.
            eedf_func (callable): EEDF function.

        Returns:
            dict: Dictionary mapping level IDs to their calculated populations in cm^-3.
        """
        # Initialize Rate Matrix A and Source Vector b
        A = np.zeros((self.num_levels, self.num_levels))
        b = np.zeros(self.num_levels)

        # Pre-calculate the mean 4s threshold for photorecombination cross-sections
        eps_4s_list = [self.db.query('ionization', lower_level=lvl)['ionization_energy'] for lvl in [2, 3, 4, 5]]
        eps_4s_mean = np.mean(eps_4s_list)

        # ---------------------------------------------------------
        # 1-3. GROUND STATE, CONTINUUM, AND QUENCHING
        # ---------------------------------------------------------
        # Iterate over active levels individually to assemble source terms and diagonal losses
        for active_lvl in self.active_levels:
            idx = self.idx_map[active_lvl]

            # Assign to strictly mapped variables (m=1 is always ground state)
            m = 1
            n = active_lvl

            C_nm, C_mn = self._calc_electron_excitation(m, n, Te_eV, eedf_func)
            K_nm, K_mn = self._calc_atom_excitation(m, n, Tg_K)
            A_mn, Lambda_mn = self._calc_radiative_rates(n, m, n_1, R_cm, Tg_K)

            # Sum of all loss processes from level n back to the ground state (n -> m)
            loss_to_ground = (n_e * C_mn) + (n_1 * K_mn) + (Lambda_mn * A_mn)
            # Sum of all source processes from ground state up to level n (m -> n)
            source_from_ground = n_1 * ((n_e * C_nm) + (n_1 * K_nm))

            # Continuum Interactions
            S_n, O_n, V_n, W_n = self._calc_ionization_rates(n, Te_eV, Tg_K, eedf_func)
            R_n = self._calc_photorecombination(n, Te_eV, eedf_func, eps_4s_mean=eps_4s_mean)

            loss_to_cont = (n_e * S_n) + (n_1 * V_n)

            core_list = self.db.query('levels', lower_level=n).get('qn_core', [])
            dominant_core = core_list[0] if core_list else None
            n_plus = n_ion * (1.0 / 3.0 if dominant_core == '1/2' else 2.0 / 3.0)

            source_from_cont = n_e * n_plus * ((n_e * O_n) + (n_1 * W_n) + R_n)

            # Metastable Quenching
            loss_diff, loss_3b = self._calc_quenching_rates(n, Tg_K, n_1, R_cm)
            loss_quenching = loss_diff + loss_3b

            # Initialize Diagonal Elements and Source Vector
            A[idx, idx] = -(loss_to_ground + loss_to_cont + loss_quenching)
            b[idx] = -(source_from_ground + source_from_cont)

        # ---------------------------------------------------------
        # 4. INTERNAL EXCITED STATE TRANSITIONS (m <-> n)
        # ---------------------------------------------------------
        # Iterate over all unique pairs of active levels exactly ONCE
        for i, lvl_a in enumerate(self.active_levels):
            for j, lvl_b in enumerate(self.active_levels[i + 1:], start=i + 1):
                E_a = self.db.query('levels', lower_level=lvl_a)['excitation_energy']
                E_b = self.db.query('levels', lower_level=lvl_b)['excitation_energy']

                # Strictly map m to the lower energy state and n to the higher energy state
                if E_a < E_b:
                    m, n = lvl_a, lvl_b
                    idx_m, idx_n = i, j
                else:
                    m, n = lvl_b, lvl_a
                    idx_m, idx_n = j, i

                # Calculate rates for the pair exactly once
                C_nm, C_mn = self._calc_electron_excitation(m, n, Te_eV, eedf_func)
                K_nm, K_mn = self._calc_atom_excitation(m, n, Tg_K)
                A_mn, Lambda_mn = self._calc_radiative_rates(n, m, n_1, R_cm, Tg_K)

                # Total transfer rates between the states
                rate_m_to_n = (n_e * C_nm) + (n_1 * K_nm)
                rate_n_to_m = (n_e * C_mn) + (n_1 * K_mn) + (Lambda_mn * A_mn)

                # Populate off-diagonals (Gain to destination state)
                A[idx_n, idx_m] += rate_m_to_n
                A[idx_m, idx_n] += rate_n_to_m

                # Subtract from diagonals (Loss from source state)
                A[idx_m, idx_m] -= rate_m_to_n
                A[idx_n, idx_n] -= rate_n_to_m

        populations = self.solver.solve(A, b)
        return {lvl: populations[i] for lvl, i in self.idx_map.items()}