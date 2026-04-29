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


class FluxAnalyzer:
    """
    On-Demand Flux Analyzer for the Collisional-Radiative Model.
    
    This class re-calculates transition rates for specific energy levels to provide 
    a detailed breakdown of population and depopulation mechanisms. This is 
    extremely useful for debugging the CR matrix and understanding the physical 
    dominance of different processes under specific plasma conditions.
    """

    def __init__(self, engine, populations, Te_eV, Tg_K, n_e, n_1, R_cm, eedf_func):
        """
        Initializes the FluxAnalyzer with simulation results and plasma parameters.

        Parameters:
            engine (PhysicsEngine): The physics engine instance used for the simulation.
            populations (dict): Calculated population densities [cm^-3].
            Te_eV (float): Electron temperature in eV.
            Tg_K (float): Gas temperature in K.
            n_e (float): Electron density in cm^-3.
            n_1 (float): Ground state density in cm^-3.
            R_cm (float): Plasma radius in cm.
            eedf_func (callable): EEDF function.
        """
        self.engine = engine
        self.populations = populations
        self.Te_eV = Te_eV
        self.Tg_K = Tg_K
        self.n_e = n_e
        self.n_1 = n_1
        self.R_cm = R_cm
        self.eedf_func = eedf_func

    def _get_population(self, level):
        """Safely fetches population density for a level, handling ground state (level 1)."""
        if level == 1:
            return self.n_1
        return self.populations.get(level, 0.0)

    def analyze_level(self, p, total_levels=65):
        """
        Calculates all flux terms (rates and frequencies) for a specific level.

        Separates fluxes into internal ladder transitions (upward/downward) 
        and continuum interactions (ionization/recombination).

        Parameters:
            p (int): The ID of the level to analyze.
            total_levels (int): Total number of levels in the model.

        Returns:
            dict: Structured data containing all calculated flux components.
        """

        report = {
            'level': p,
            'depopulation_freq': {'up': {'e': 0.0, 'a': 0.0}, 'down': {'rad': 0.0, 'e': 0.0, 'a': 0.0}},
            'population_rate': {'from_up': {'rad': 0.0, 'e': 0.0, 'a': 0.0}, 'from_down': {'e': 0.0, 'a': 0.0}},
            'continuum': {
                'depopulation_freq': {'e': 0.0, 'a': 0.0},
                'population_rate': {'e': 0.0, 'a': 0.0, 'rad': 0.0}
            }
        }

        E_p = self.engine.db.query('levels', lower_level=p)['excitation_energy']

        # =====================================================================
        # 1. INTERNAL LADDER INTERACTIONS (Level p <-> Level i)
        # =====================================================================
        for i in range(1, total_levels + 1):
            if i == p: continue

            E_i = self.engine.db.query('levels', lower_level=i)['excitation_energy']
            pop_i = self._get_population(i)

            # Re-calculate transition rates between p and i
            # Returns (excitation m->n, de-excitation n->m)
            if E_i > E_p:
                # p is lower level (m), i is upper level (n)
                C_pi, F_ip = self.engine._calc_electron_excitation(p, i, self.Te_eV, self.eedf_func)
                K_pi, L_ip = self.engine._calc_atom_excitation(p, i, self.Tg_K)
                
                # Upward transitions from p (Depopulation)
                report['depopulation_freq']['up']['e'] += self.n_e * C_pi
                report['depopulation_freq']['up']['a'] += self.n_1 * K_pi

                # Downward transitions from i to p (Population)
                A_pi, lam_pi = self.engine._calc_radiative_rates(p, i, self.n_1, self.R_cm, self.Tg_K)
                report['population_rate']['from_up']['rad'] += pop_i * lam_pi * A_pi
                report['population_rate']['from_up']['e'] += pop_i * self.n_e * F_ip
                report['population_rate']['from_up']['a'] += pop_i * self.n_1 * L_ip
            else:
                # i is lower level (m), p is upper level (n)
                C_ip, F_pi = self.engine._calc_electron_excitation(i, p, self.Te_eV, self.eedf_func)
                K_ip, L_pi = self.engine._calc_atom_excitation(i, p, self.Tg_K)

                # Downward transitions from p (Depopulation)
                A_ip, lam_ip = self.engine._calc_radiative_rates(i, p, self.n_1, self.R_cm, self.Tg_K)
                report['depopulation_freq']['down']['rad'] += lam_ip * A_ip
                report['depopulation_freq']['down']['e'] += self.n_e * F_pi
                report['depopulation_freq']['down']['a'] += self.n_1 * L_pi

                # Upward transitions from i to p (Population)
                report['population_rate']['from_down']['e'] += pop_i * self.n_e * C_ip
                report['population_rate']['from_down']['a'] += pop_i * self.n_1 * K_ip

        # =====================================================================
        # 2. CONTINUUM INTERACTIONS (Level p <-> Ionization)
        # =====================================================================
        eps_4s_list = [self.engine.db.query('ionization', lower_level=lvl)['ionization_energy'] for lvl in [2, 3, 4, 5]]
        eps_4s_mean = float(np.mean(eps_4s_list))

        # Re-calculate ionization and recombination rates
        S_p, O_p, V_p, W_p = self.engine._calc_ionization_rates(p, self.Te_eV, self.Tg_K, self.eedf_func)
        R_p = self.engine._calc_photorecombination(p, self.Te_eV, self.eedf_func, eps_4s_mean=eps_4s_mean)

        core_p_list = self.engine.db.query('levels', lower_level=p).get('qn_core', [])
        dominant_core = core_p_list[0] if core_p_list else None
        n_plus = self.n_e * (1.0 / 3.0 if dominant_core == '1/2' else 2.0 / 3.0)

        # Depopulation Frequencies (s^-1)
        report['continuum']['depopulation_freq']['e'] = self.n_e * S_p
        report['continuum']['depopulation_freq']['a'] = self.n_1 * V_p

        # Population Source Rates (cm^-3 s^-1)
        report['continuum']['population_rate']['e'] = self.n_e * n_plus * self.n_e * O_p
        report['continuum']['population_rate']['a'] = self.n_e * n_plus * self.n_1 * W_p
        report['continuum']['population_rate']['rad'] = self.n_e * n_plus * R_p

        return report

    def print_report(self, p, total_levels=65):
        """
        Generates and prints a formatted diagnostic report for a specific level.
        """
        data = self.analyze_level(p, total_levels)

        print(f"\n{'=' * 60}")
        print(f" FLUX DIAGNOSTICS FOR LEVEL: {p}")
        print(f" (Te = {self.Te_eV} eV, ne = {self.n_e:.1E} cm^-3, ng = {self.n_1:.1E} cm^-3)")
        print(f"{'=' * 60}")

        print("\n[ INTERNAL LADDER: DEPOPULATION FREQUENCIES (s^-1) ]")
        print("  UPWARD (To E_i > E_p):")
        print(f"    Electron Excitation: {data['depopulation_freq']['up']['e']:.2E}")
        print(f"    Atom Excitation:     {data['depopulation_freq']['up']['a']:.2E}")

        print("\n  DOWNWARD (To E_i < E_p):")
        print(f"    Spontaneous Em.:     {data['depopulation_freq']['down']['rad']:.2E}")
        print(f"    Electron De-Exc.:    {data['depopulation_freq']['down']['e']:.2E}")
        print(f"    Atom De-Exc.:        {data['depopulation_freq']['down']['a']:.2E}")

        print("\n[ INTERNAL LADDER: POPULATION RATES (cm^-3 s^-1) ]")
        print("  FROM UPPER (Cascading Down):")
        print(f"    Spontaneous Em.:     {data['population_rate']['from_up']['rad']:.2E}")
        print(f"    Electron De-Exc.:    {data['population_rate']['from_up']['e']:.2E}")
        print(f"    Atom De-Exc.:        {data['population_rate']['from_up']['a']:.2E}")

        print("\n  FROM LOWER (Pumping Up):")
        print(f"    Electron Excitation: {data['population_rate']['from_down']['e']:.2E}")
        print(f"    Atom Excitation:     {data['population_rate']['from_down']['a']:.2E}")

        print("\n[ CONTINUUM INTERACTIONS (Ionization / Recomb) ]")
        print("  DEPOPULATION TO CONTINUUM (Freq: s^-1):")
        print(f"    Electron Ionization: {data['continuum']['depopulation_freq']['e']:.2E}")
        print(f"    Atom Ionization:     {data['continuum']['depopulation_freq']['a']:.2E}")

        print("\n  POPULATION FROM CONTINUUM (Rate: cm^-3 s^-1):")
        print(f"    3-Body Electron Rec: {data['continuum']['population_rate']['e']:.2E}")
        print(f"    3-Body Atom Rec:     {data['continuum']['population_rate']['a']:.2E}")
        print(f"    Radiative Recomb:    {data['continuum']['population_rate']['rad']:.2E}")
        print(f"{'=' * 60}\n")