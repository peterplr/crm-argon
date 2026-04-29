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
import re
from pathlib import Path
import math


class CRDataLoader:
    """
    Extract Phase: Handles all file I/O to load raw CSV data into memory.
    Ensures all necessary data files are located in the expected directory structure.
    """

    def __init__(self, script_dir):
        """
        Initializes the loader with the base directory of the data preparation scripts.

        Args:
            script_dir (str or Path): The root directory where 'raw' data is stored.
        """
        self.script_dir = Path(script_dir)
        self.raw_dir = self.script_dir / 'raw'

        # Dictionary mapping logical names to their physical file paths
        self.paths = {
            'vlcek_levels': self.raw_dir / 'Vlcek' / 'levels_and_params.csv',
            'vlcek_photo': self.raw_dir / 'Vlcek' / 'photoionization.csv',
            'nist': self.raw_dir / 'NIST' / 'nist_argon_raw.csv',
            'kimura_parameters': self.raw_dir / 'Kimura' / 'kimura_parameters.csv',
            'kimura_einstein': self.raw_dir / 'Kimura' / 'kimura_einstein.csv',
            'vlcek_spin_forb': self.raw_dir / 'Vlcek' / 'inter_state_spin_forbidden.csv',
            'tachibana': self.raw_dir / 'Tachibana' / 'constants_metastables.csv',
            'katsonis': self.raw_dir / 'Katsonis' / 'katsonis_photoionization.csv'
        }

    def load_all_raw(self):
        """
        Loads all CSV files defined in self.paths into a dictionary of dictionaries.

        Returns:
            dict: A collection of raw data tables, keyed by their logical names.
        """
        print("=== Initializing Data Loader ===")
        raw_data = {
            'vlcek_levels': self._read_csv_dict(self.paths['vlcek_levels']),
            'vlcek_photo': self._read_csv_dict(self.paths['vlcek_photo']),
            'nist': self._read_csv_dict(self.paths['nist']),  # Now a dictionary!
            'kimura_parameters': self._read_csv_dict(self.paths['kimura_parameters']),
            'kimura_einstein': self._read_csv_dict(self.paths['kimura_einstein']),
            'vlcek_spin_forb': self._read_csv_dict(self.paths['vlcek_spin_forb']),
            'tachibana': self._read_csv_dict(self.paths['tachibana']),
            'katsonis_photo': self._read_csv_dict(self.paths['katsonis']),
        }
        print("=== Raw Data Successfully Loaded ===\n")
        return raw_data

    def _read_csv_dict(self, path):
        """
        Helper method to read a CSV file into a list of dictionaries.

        Args:
            path (Path): Path to the CSV file.

        Returns:
            list: List of row dictionaries, or an empty list if the file is missing.
        """
        if not path.exists():
            print(f"  [!] WARNING: Could not find {path}")
            return []
        with open(path, mode='r', encoding='utf-8') as f:
            return list(csv.DictReader(f))


class CRDataProcessor:
    """
    Transform Phase: Centralized data processing engine.
    """

    def __init__(self, raw_data):
        self.raw_data = raw_data

        # --- Internal Master State ---
        self.levels_db = {}

        # --- Einstein A and f_ik Parameters ---
        self.nist_covered_multiplets = set()
        self.nist_multiplet_fiks = {}
        self.nist_fine_db = []
        self.kimura_einstein_raw_db = {}

        self.einstein_nist_db = {}
        self.einstein_kimura_db = {}
        self.einstein_hydrogenic_db = {}

        # --- Electron Excitation Transition Databases ---
        self.allowed_ground_db = {}
        self.allowed_kimura_db = {}
        self.parity_forbidden_db = {}
        self.spin_forbidden_db = {}
        self.ionization_db = {}
        self.photoionization_db = {}
        self.metastable_quenching_db = {}

        # Constants
        self.E_ION_UNPRIMED = 15.7596119
        self.E_ION_PRIMED = 15.9371056

        # ==========================================

    # HELPER METHODS
    # ==========================================
    @staticmethod
    def _categorize_transition(qn_lower, qn_upper):
        core_m = qn_lower['core']
        core_n = qn_upper['core']
        core_changes = (core_m != core_n)

        l_m = qn_lower['l']
        l_n = qn_upper['l']

        if l_m == "all" or l_n == "all":
            delta_l_is_one = True
            delta_l_is_zero = False
        else:
            delta_l = abs(int(l_m) - int(l_n))
            delta_l_is_one = (delta_l == 1)
            delta_l_is_zero = (delta_l == 0)

        j_m = qn_lower['J']
        j_n = qn_upper['J']

        if j_m == "all" or j_n == "all":
            j_valid_for_dipole = True
        else:
            j_m, j_n = int(j_m), int(j_n)
            delta_j = abs(j_m - j_n)
            j_valid_for_dipole = (delta_j <= 1) and not (j_m == 0 and j_n == 0)

        is_dipole_allowed = delta_l_is_one and j_valid_for_dipole

        if core_changes:
            return 'S' if is_dipole_allowed else 'Ignore'
        else:
            if is_dipole_allowed:
                return 'A'
            elif delta_l_is_zero:
                return 'P'
            else:
                return 'Ignore'

    def _parse_quantum_numbers(self, designation, energy):
        qns = []
        if designation == "3p6":
            return [{"n": 3, "l": 1, "K": "all", "J": 0, "core": "3/2"}]

        match_npqn = re.search(r'n_pqn\s*=\s*(\d+)(\'?)', designation)
        if match_npqn:
            n_val = int(match_npqn.group(1))
            is_prime = bool(match_npqn.group(2))
            core = "1/2" if (is_prime or energy > self.E_ION_UNPRIMED) else "3/2"
            return [{"n": n_val, "l": "all", "K": "all", "J": "all", "core": core}]

        parts = [p.strip() for p in designation.split('+')]
        for part in parts:
            core = "1/2" if ("'" in part or energy > self.E_ION_UNPRIMED) else "3/2"

            if '...' in part:
                match_n = re.search(r'^(\d+)', part)
                if match_n:
                    n_val = int(match_n.group(1))
                    match_l = re.search(r'([spdfghi])', part.lower())
                    start_l = "spdfghi".find(match_l.group(1)) if match_l else 0
                    for l_val in range(start_l, n_val):
                        qns.append({"n": n_val, "l": l_val, "K": "all", "J": "all", "core": core})
                continue

            if '[' in part:
                match_nl = re.search(r'^(\d+)([spdfghi])', part.lower())
                if match_nl:
                    n_val = int(match_nl.group(1))
                    l_val = "spdfghi".find(match_nl.group(2))
                    blocks = re.findall(r'\[(.*?)]([\d,]*)', part)
                    for k_val, j_vals in blocks:
                        js = [int(j) for j in j_vals.split(',') if j.strip()]
                        if not js:
                            qns.append({"n": n_val, "l": l_val, "K": k_val, "J": "all", "core": core})
                        else:
                            for j in js:
                                qns.append({"n": n_val, "l": l_val, "K": k_val, "J": j, "core": core})
                continue

            sub_parts = [s.strip() for s in part.split(',')]
            primary_n_match = re.search(r'^(\d+)', sub_parts[0])

            if primary_n_match:
                n_val = int(primary_n_match.group(1))
                for sub in sub_parts:
                    l_match = re.search(r'([spdfghi])', sub.lower())
                    if l_match:
                        l_val = "spdfghi".find(l_match.group(1))
                        qns.append({"n": n_val, "l": l_val, "K": "all", "J": "all", "core": core})

        return qns

    def _clean_nist_value(self, val):
        if val is None: return ""
        return val.replace('="', '').replace('"', '').strip()

    def _parse_nist_to_qns(self, conf, term, j_str):
        if "3p6" in conf and "1S" in term and j_str == "0":
            return {"n": 3, "l": 1, "K": "all", "J": 0, "core": "3/2"}

        shell = conf.split('.')[-1]
        match_n = re.search(r'(\d+)', shell)
        match_l = re.search(r'([spdfghi])', shell.lower())

        if not match_n or not match_l:
            print(
                f"  [!] WARNING: Expected valid shell format (numeric n, letter l) in NIST conf '{conf}', received shell '{shell}'.")
            return None

        n_val = int(match_n.group(1))
        l_val = "spdfghi".find(match_l.group(1))
        core = "1/2" if "<1/2>" in conf else "3/2"
        match_k = re.search(r'\[(.*?)]', term)
        k_val = match_k.group(1) if match_k else "all"
        j_val = int(j_str) if j_str.isdigit() else "all"

        return {"n": n_val, "l": l_val, "K": k_val, "J": j_val, "core": core}

    def _map_nist_to_vlcek(self, nist_qns):
        if not nist_qns:
            return None

        for vlcek_id, data in self.levels_db.items():
            for qn in data['quantum_numbers']:
                match_n = (qn['n'] == nist_qns['n'])
                match_l = (qn['l'] == "all" or qn['l'] == nist_qns['l'])
                match_c = (qn['core'] == nist_qns['core'])
                match_k = (qn['K'] == "all" or str(qn['K']) == str(nist_qns['K']))
                match_j = (qn['J'] == "all" or str(qn['J']) == str(nist_qns['J']))

                if match_n and match_l and match_c and match_k and match_j:
                    return vlcek_id

        print(
            f"  [!] WARNING: Expected to map NIST quantum numbers to a Vlcek level, but no match was found. Received QNs: {nist_qns}")
        return None

    def _match_kimura_and_vlcek(self, kimura_shell, target_core):
        """
        Searches the master levels_db to find all Vlcek Level IDs that
        contain the specified Kimura shell.
        """
        matches = []

        # Clean the Kimura shell just in case it contains a rogue prime
        clean_kimura_shell = kimura_shell.replace("'", "")

        m_shell = re.match(r'^(\d+)', clean_kimura_shell)
        n_kimura = int(m_shell.group(1)) if m_shell else None

        for vlcek_id, data in self.levels_db.items():
            for qn in data['quantum_numbers']:

                # 1. First, strictly filter by the core. This makes string primes irrelevant.
                if qn['core'] != target_core:
                    continue

                match_found = False

                # 2. Check for lumped n states (e.g., n_pqn = 10)
                if qn['l'] == "all":
                    if qn['n'] == n_kimura:
                        match_found = True

                # 3. Check for explicitly defined orbitals (e.g., 4p)
                else:
                    l_chars = "spdfghiklmno"
                    if int(qn['l']) < len(l_chars):
                        # Construct a clean shell string (e.g., "4p") without primes
                        shell_str = f"{qn['n']}{l_chars[int(qn['l'])]}"
                        if shell_str == clean_kimura_shell:
                            match_found = True

                if match_found and vlcek_id not in matches:
                    matches.append(vlcek_id)

        return matches

    # ==========================================
    # DATA PROCESSING PIPELINE
    # ==========================================
    def process_all(self):
        print("=== Processing Data ===")
        self.nist_covered_multiplets = set()

        self.process_vlcek_levels()
        self.process_nist_data()
        self.process_kimura_einstein_data()
        self.process_hydrogenic_einstein_data()
        self.process_allowed_data()
        self.process_parity_forbidden_data()
        self.process_spin_forbidden_data()
        self.process_photoionization_data()
        self.process_metastable_quenching_data()
        print("=== Processing Complete ===\n")

    def process_vlcek_levels(self):
        print("Processing Vlcek levels and extracting ground state transitions...")
        for row in self.raw_data['vlcek_levels']:
            n_level = int(row['n'])
            energy = float(row['energy_ev'])
            designation = row['designation']

            g_val = row.get('g', '').strip()
            if not g_val:
                print(
                    f"  [!] WARNING: Expected statistical weight (g) for Vlcek level {n_level}, received empty string. Defaulting to 1.0.")
                g_val = 1.0
            else:
                g_val = float(g_val)

            qns = self._parse_quantum_numbers(designation, energy)

            self.levels_db[n_level] = {
                'designation': designation,
                'excitation_energy': energy,
                'g': g_val,
                'quantum_numbers': qns,
                'raw_row': row
            }

            t_type = row.get('transition_type', '').strip()
            if t_type:
                alpha_str = row.get('alpha', '').strip()
                beta_str = row.get('beta', '').strip()
                alpha_val = float(alpha_str) if alpha_str else 0.0
                beta_val = float(beta_str) if beta_str else 1.0

                trans_key = (1, n_level)
                if t_type == 'A':
                    self.allowed_ground_db[trans_key] = {'excitation_energy': energy, 'alpha': alpha_val,
                                                         'beta': beta_val}
                elif t_type == 'P':
                    self.parity_forbidden_db[trans_key] = {'excitation_energy': energy, 'alpha': alpha_val}
                elif t_type == 'S':
                    self.spin_forbidden_db[trans_key] = {'excitation_energy': energy, 'alpha': alpha_val, 'equation': 2}

            xi_str = row.get('xi_ion', '').strip()
            alpha_ion_str = row.get('alpha_ion', '').strip()
            beta_ion_str = row.get('beta_ion', '').strip()

            if xi_str and alpha_ion_str and beta_ion_str:
                core_val = qns[0]['core']
                limit = self.E_ION_PRIMED if core_val == "1/2" else self.E_ION_UNPRIMED
                self.ionization_db[n_level] = {
                    'ionization_energy': limit - energy,
                    'xi_ion': int(float(xi_str)),
                    'alpha_ion': float(alpha_ion_str),
                    'beta_ion': float(beta_ion_str)
                }

    def process_nist_data(self):
        """
        Extracts and maps NIST Einstein A and f_ik parameters.
        Separates fine-structured data tracking from multiplet coverage logic.
        Aggregates lumped Vlcek capacities and immediately performs statistical
        averaging, so the resulting database contains final A and f values for export.
        """
        print("Processing NIST Einstein data...")

        # Temporary dictionary to hold summed capacities before statistical averaging
        lumped_sums = {}

        for row_idx, row in enumerate(self.raw_data['nist']):
            aki_raw = self._clean_nist_value(row.get('Aki(s^-1)', ''))
            fik_raw = self._clean_nist_value(row.get('fik', ''))

            if not aki_raw and not fik_raw:
                print(f"  [i] INFO: Skipping NIST row {row_idx} - Neither Aki nor fik values are present.")
                continue

            conf_i, term_i, j_i = (self._clean_nist_value(row.get('conf_i', '')), self._clean_nist_value(row.get('term_i', '')),
                                   self._clean_nist_value(row.get('J_i', '')))
            conf_k, term_k, j_k = (self._clean_nist_value(row.get('conf_k', '')), self._clean_nist_value(row.get('term_k', '')),
                                   self._clean_nist_value(row.get('J_k', '')))

            if not conf_i or not conf_k:
                print(f"  [!] WARNING: Skipping NIST row {row_idx} - Incomplete configuration data (conf_i: '{conf_i}', conf_k: '{conf_k}').")
                continue

            qns_i, qns_k = self._parse_nist_to_qns(conf_i, term_i, j_i), self._parse_nist_to_qns(conf_k, term_k, j_k)
            if not qns_i or not qns_k:
                print(f"  [!] WARNING: Skipping NIST row {row_idx} - Failed to parse quantum numbers from configurations.")
                continue

            vlcek_id_i, vlcek_id_k = self._map_nist_to_vlcek(qns_i), self._map_nist_to_vlcek(qns_k)
            if not vlcek_id_i or not vlcek_id_k:
                print(f"  [!] WARNING: Skipping NIST row {row_idx} - NIST states failed to map to Vlcek levels (i: {vlcek_id_i}, k: {vlcek_id_k}).")
                continue

            # Safely extract statistical weights and catch un-castable data
            g_i_raw, g_k_raw = self._clean_nist_value(row.get('g_i', '')), self._clean_nist_value(row.get('g_k', ''))
            try:
                g_i = float(g_i_raw) if g_i_raw else self.levels_db[vlcek_id_i]['g']
                g_k = float(g_k_raw) if g_k_raw else self.levels_db[vlcek_id_k]['g']
            except ValueError as e:
                print(f"  [!] CRITICAL ERROR: Skipping NIST row {row_idx} - Malformed statistical weight (g) encountered: {e}")
                continue

            # Standardize ordering so 'lower' strictly represents the lower energy state
            if self.levels_db[vlcek_id_i]['excitation_energy'] < self.levels_db[vlcek_id_k]['excitation_energy']:
                lvl_m, lvl_n = vlcek_id_i, vlcek_id_k
                qns_lower, qns_upper = qns_i, qns_k
                core_lower, core_upper = qns_i['core'], qns_k['core']
                g_lower, g_upper = g_i, g_k
            else:
                lvl_m, lvl_n = vlcek_id_k, vlcek_id_i
                qns_lower, qns_upper = qns_k, qns_i
                core_lower, core_upper = qns_k['core'], qns_i['core']
                g_lower, g_upper = g_k, g_i

            # --- Fine-Structured Database (Line-by-Line Saving) ---
            l_chars = "spdfghiklmno"
            shell_lower = f"{qns_lower['n']}{l_chars[int(qns_lower['l'])]}" if qns_lower['l'] != "all" else None
            shell_upper = f"{qns_upper['n']}{l_chars[int(qns_upper['l'])]}" if qns_upper['l'] != "all" else None

            fine_entry = {
                'raw_row_idx': row_idx,
                'qns_lower': qns_lower,
                'qns_upper': qns_upper,
                'shell_lower': shell_lower,
                'shell_upper': shell_upper,
                'g_lower': g_lower,
                'g_upper': g_upper,
                'A': float(aki_raw) if aki_raw else 0.0,
                'f': float(fik_raw) if fik_raw else 0.0
            }
            # Save every row without additive lumping
            self.nist_fine_db.append(fine_entry)

            # --- Multiplet Coverage Tracking ---
            if shell_lower and shell_upper and shell_lower != "3p" and shell_upper != "3p":
                # Add to Veto Sets for Kimura Processing
                self.nist_covered_multiplets.add((shell_lower, shell_upper, core_lower))
                self.nist_covered_multiplets.add((shell_lower, shell_upper, core_upper))

                # Aggregate multiplet fractions needed later for Kimura Allowed branch A
                if fik_raw:
                    multiplet_key_no_core = (shell_lower, shell_upper)
                    self.nist_multiplet_fiks[multiplet_key_no_core] = self.nist_multiplet_fiks.get(multiplet_key_no_core, 0.0) + float(fik_raw)

            # --- Accumulate Lumped Vlcek Capacities ---
            trans_key = (lvl_m, lvl_n)
            if trans_key not in lumped_sums:
                lumped_sums[trans_key] = {'gA': 0.0, 'gf': 0.0}

            if aki_raw:
                lumped_sums[trans_key]['gA'] += float(aki_raw) * g_upper
            if fik_raw:
                lumped_sums[trans_key]['gf'] += float(fik_raw) * g_lower

        # --- Final Processing: Statistical Averaging ---
        print("  Finalizing NIST Einstein averages for mapped Vlcek transitions...")
        for trans_key, capacities in lumped_sums.items():
            m, n = trans_key
            g_lumped_m = self.levels_db[m]['g']
            g_lumped_n = self.levels_db[n]['g']

            if g_lumped_m <= 0 or g_lumped_n <= 0:
                raise ValueError(f"CRITICAL: Invalid statistical weight (g<=0) for Vlcek levels {m} or {n}. Cannot perform division.")

            # Resolve the statistical average directly inside the processor
            self.einstein_nist_db[trans_key] = {
                'A': capacities['gA'] / g_lumped_n,
                'f': capacities['gf'] / g_lumped_m
            }

    def distribute_multiplet(self, macro_value, target_level_index, target_core, target_l):
        """
        Distributes a macroscopic transition parameter to a specific target level
        by dynamically calculating theoretical statistical weights and safely
        extracting only the matching components from mixed/lumped states.
        """
        level_data = self.levels_db[target_level_index]

        # Calculate the Theoretical Denominator (Total g of the target multiplet)
        jc = 1.5 if str(target_core).strip() == "3/2" else 0.5
        s = 0.5  # Electron spin
        g_macro_total = (2.0 * jc + 1.0) * (2.0 * target_l + 1.0) * (2.0 * s + 1.0)

        # Calculate the True Numerator (Effective g of the specific target components)
        eff_g_target = 0.0

        # Iterate through all quantum number components comprising this lumped level
        for qn in level_data.get('quantum_numbers', []):
            l_val_str = str(qn.get('l', '')).strip().lower()

            # Match if the level explicitly has our target_l, OR if it lumps 'all' l's together
            if l_val_str == str(target_l) or l_val_str == 'all':
                j_val_str = str(qn.get('J', '')).strip().lower()

                # If J is a specific number, g = 2J + 1
                if j_val_str != 'all':
                    eff_g_target += (2.0 * float(j_val_str) + 1.0)
                else:
                    # If J is 'all', it represents the entire subshell for that core and l
                    eff_g_target += g_macro_total

        # Prevent division by zero if this level actually contains NO valid target states
        if eff_g_target == 0:
            return 0.0

        # Calculate the true branching fraction and distribute
        fraction = eff_g_target / g_macro_total

        return macro_value * fraction

    def process_kimura_einstein_data(self):
        """
        Extracts Kimura theoretical Einstein A parameters and calculates multiplet f values.
        Populates the raw Kimura database for reference and redistributes the values
        onto the Vlcek mapping for any multiplets not already covered by NIST data.
        """
        print("Processing Kimura Einstein data...")

        vetoed_multiplets = []
        added_multiplets = []

        for row_idx, row in enumerate(self.raw_data['kimura_einstein']):
            lower_shell = row.get('lower_shell', '').strip()
            upper_shell = row.get('upper_shell', '').strip()

            if not lower_shell or not upper_shell:
                print(f"  [!] WARNING: Skipping Kimura row {row_idx} - Missing shell definitions.")
                continue

            # Identify orbital angular momentum (l) from shell strings
            l_chars = "spdfghiklmno"
            lower_l_char = lower_shell[-1].lower()
            upper_l_char = upper_shell[-1].lower()

            if lower_l_char not in l_chars or upper_l_char not in l_chars:
                print(
                    f"  [!] WARNING: Skipping Kimura row {row_idx} - Invalid orbital character in shells ('{lower_l_char}' or '{upper_l_char}').")
                continue

            l_lower_val = l_chars.find(lower_l_char)
            l_upper_val = l_chars.find(upper_l_char)

            # Process both cores independently
            for core, col_name in [("3/2", 'A_2P3_2'), ("1/2", 'A_2P1_2')]:
                a_raw = row.get(col_name, '').strip()
                if not a_raw:
                    continue

                try:
                    A_val = float(a_raw)
                except ValueError as e:
                    print(f"  [!] CRITICAL ERROR: Skipping Kimura core {core} row {row_idx} - Malformed A value: {e}")
                    continue

                # Theoretical statistical capacities of the macro states
                core_g = 4 if core == "3/2" else 2
                g_lower_kimura = core_g * 2 * (2 * l_lower_val + 1)
                g_upper_kimura = core_g * 2 * (2 * l_upper_val + 1)

                lower_matches = self._match_kimura_and_vlcek(lower_shell, core)
                upper_matches = self._match_kimura_and_vlcek(upper_shell, core)

                if not lower_matches or not upper_matches:
                    print(
                        f"  [i] INFO: Kimura {lower_shell} -> {upper_shell} (Core {core}) has no valid Vlcek mapping. Skipping distribution.")
                    continue

                # Energy averaging using Vlcek lumped g to determine transition wavelength
                g_lower_vlcek_total = sum(self.levels_db[m]['g'] for m in lower_matches)
                g_upper_vlcek_total = sum(self.levels_db[n]['g'] for n in upper_matches)

                e_lower_avg = sum(self.levels_db[m]['excitation_energy'] * self.levels_db[m]['g'] for m in
                                  lower_matches) / g_lower_vlcek_total
                e_upper_avg = sum(self.levels_db[n]['excitation_energy'] * self.levels_db[n]['g'] for n in
                                  upper_matches) / g_upper_vlcek_total
                delta_E = abs(e_upper_avg - e_lower_avg)

                # Calculate oscillator strength (f) for the macro transition
                if delta_E > 0:
                    f_val = 2.3046e-8 * (A_val / delta_E**2) * (g_upper_kimura / g_lower_kimura)
                else:
                    print(
                        f"  [!] WARNING: Delta E is zero or negative for {lower_shell} -> {upper_shell}. Defaulting f to 0.0.")
                    f_val = 0.0

                # Save to the raw, unmapped macro database
                raw_key = (lower_shell, upper_shell, core)
                self.kimura_einstein_raw_db[raw_key] = {
                    'A': A_val,
                    'f': f_val,
                    'g_lower_kimura': g_lower_kimura,
                    'g_upper_kimura': g_upper_kimura
                }

                # Multiplet Veto Logic: Skip mapping to Vlcek if NIST already covers this transition
                if (lower_shell, upper_shell, core) in self.nist_covered_multiplets:
                    vetoed_multiplets.append(f"{lower_shell} -> {upper_shell} (Core {core})")
                    continue

                added_multiplets.append(f"{lower_shell} -> {upper_shell} (Core {core})")

                # Statistical redistribution onto Vlcek lumped states
                for m in lower_matches:
                    for n in upper_matches:
                        # Establish proper transition direction
                        vlcek_upper = max(m, n, key=lambda x: self.levels_db[x]['excitation_energy'])
                        vlcek_lower = min(m, n, key=lambda x: self.levels_db[x]['excitation_energy'])

                        if self.levels_db[vlcek_lower]['excitation_energy'] >= self.levels_db[vlcek_upper][
                            'excitation_energy']:
                            continue

                        # Distribute A (Downward transition -> Target is the lower state)
                        specific_A = self.distribute_multiplet(A_val, vlcek_lower, core, l_lower_val)

                        # Distribute f (Upward transition -> Target is the upper state)
                        specific_f = self.distribute_multiplet(f_val, vlcek_upper, core, l_upper_val)

                        vlcek_key = (vlcek_lower, vlcek_upper)

                        if vlcek_key not in self.einstein_kimura_db:
                            self.einstein_kimura_db[vlcek_key] = {'A': 0.0, 'f': 0.0}

                        self.einstein_kimura_db[vlcek_key]['A'] += specific_A
                        self.einstein_kimura_db[vlcek_key]['f'] += specific_f

        # Supervision report on veto status
        print("\n  --- MULTIPLET VETO SUPERVISION REPORT ---")
        print(f"  [+] Added {len(added_multiplets)} missing multiplets from Kimura to Einstein export.")
        print(
            f"  [-] Safely vetoed {len(vetoed_multiplets)} Kimura multiplets from Einstein export (Already mapped by NIST).")
        print("  -----------------------------------------\n")

    def process_hydrogenic_einstein_data(self):
        """
        Calculates theoretical oscillator strengths (f) and Einstein A coefficients
        for highly excited Rydberg states (e.g., n >= 45) using the Kramers
        semiclassical hydrogenic approximation.

        Populates self.einstein_hydrogenic_db to act as a fallback where NIST
        and Kimura data do not exist.
        """
        print("Processing Hydrogenic Einstein data (n >= 45)...")

        hydrogenic_added = 0

        for m in range(45, 65):
            for n in range(m + 1, 66):
                qn_m = self.levels_db[m]['quantum_numbers'][0]
                qn_n = self.levels_db[n]['quantum_numbers'][0]

                # Do not overwrite Kimura
                if (m,n) in [(45,46),(45,47),(46,47)]:
                    continue

                # Only allow transitions between the same core state (1/2 -> 1/2 or 3/2 -> 3/2)
                if qn_m['core'] != qn_n['core']:
                    continue

                n_eff_m, n_eff_n = qn_m['n'], qn_n['n']

                if n_eff_m != n_eff_n:
                    # Semiclassical oscillator strength formula for lumped macroscopic shells
                    prefactor = 32.0 / (3.0 * math.sqrt(3) * math.pi)
                    numerator = n_eff_m * (n_eff_n ** 3)
                    denominator = (n_eff_n ** 2 - n_eff_m ** 2) ** 3

                    f_mn = prefactor * (numerator / denominator)
                else:
                    f_mn = 0.0

                if f_mn > 0:
                    lower_lvl, upper_lvl = (m, n) if self.levels_db[m]['excitation_energy'] < self.levels_db[n][
                        'excitation_energy'] else (n, m)
                    trans_key = (lower_lvl, upper_lvl)

                    # Fetch parameters to convert f into Einstein A
                    exc_energy = self.levels_db[upper_lvl]['excitation_energy'] - self.levels_db[lower_lvl][
                        'excitation_energy']
                    g_lower = self.levels_db[lower_lvl]['g']
                    g_upper = self.levels_db[upper_lvl]['g']

                    # Converted formula: A_nm = Constant * f_mn * E_eV^2 * (g_lower / g_upper)
                    A_nm = 4.3392e7 * f_mn * (exc_energy ** 2) * (g_lower / g_upper)

                    self.einstein_hydrogenic_db[trans_key] = {
                        'A': A_nm,
                        'f': f_mn
                    }
                    hydrogenic_added += 1

        # Supervision report on algorithmic generation
        print("\n  --- HYDROGENIC LADDER SUPERVISION REPORT ---")
        print(f"  [+] Algorithmically generated {hydrogenic_added} transition probabilities (A and f).")
        print("  --------------------------------------------\n")

    def process_allowed_data(self):
        """
        Processes Kimura parameters for Allowed transitions (|Delta l| = 1).
        Calculates alpha values by multiplying Kimura's K by the previously calculated f_macro.
        Routes the distribution of these values depending on whether high-fidelity NIST
        fine-structure data is available to guide the breakdown.
        """
        print("Processing Kimura Allowed transitions...")

        for row_idx, row in enumerate(self.raw_data['kimura_parameters']):
            lower_shell = row.get('lower_shell', '').strip()
            upper_shell = row.get('upper_shell', '').strip()

            if not lower_shell or not upper_shell:
                print(f"  [!] WARNING: Skipping Kimura allowed row {row_idx} - Missing shell definitions.")
                continue

            l_chars = "spdfghiklmno"
            lower_l_char = lower_shell[-1].lower()
            upper_l_char = upper_shell[-1].lower()

            if lower_l_char not in l_chars or upper_l_char not in l_chars:
                continue

            l_lower_val = l_chars.find(lower_l_char)
            l_upper_val = l_chars.find(upper_l_char)

            # Isolate only dipole-allowed transitions
            if abs(l_lower_val - l_upper_val) != 1:
                continue

            try:
                beta_val = float(row.get('beta', '1.0'))
            except ValueError:
                print(f"  [!] WARNING: Malformed beta value in allowed row {row_idx}, defaulting to 1.0.")
                beta_val = 1.0

            multiplet_key = (lower_shell, upper_shell)
            has_nist_data = multiplet_key in self.nist_multiplet_fiks and self.nist_multiplet_fiks[multiplet_key] > 0

            # Extract K values and calculate macro alphas (K * f_macro)
            alpha_macro = {}
            for core, col_name in [("3/2", 'alpha_2P3_2'), ("1/2", 'alpha_2P1_2')]:
                k_raw = row.get(col_name, '').strip()
                if not k_raw:
                    continue

                try:
                    k_val = float(k_raw)
                except ValueError:
                    print(f"  [!] CRITICAL ERROR: Malformed K value for core {core} in allowed row {row_idx}.")
                    continue

                raw_key = (lower_shell, upper_shell, core)
                if raw_key in self.kimura_einstein_raw_db:
                    f_macro = self.kimura_einstein_raw_db[raw_key]['f']
                    alpha_macro[core] = k_val * f_macro
                else:
                    print(
                        f"  [!] WARNING: Kimura allowed row {row_idx} missing corresponding f_macro data in Einstein database. Cannot calculate alpha.")

            if not alpha_macro:
                continue

            # Route the data distribution based on NIST availability
            if has_nist_data:
                # Path A: Supervised Redistribution via NIST Fine-Structure Data

                # Average the available core values to form a unified macro parameter
                val_32 = alpha_macro.get("3/2", 0.0)
                val_12 = alpha_macro.get("1/2", 0.0)
                active_cores = len(alpha_macro)
                alpha_avg = (val_32 + val_12) / active_cores if active_cores > 0 else 0.0

                sum_nist_f = self.nist_multiplet_fiks[multiplet_key]

                # Map over every saved line in the fine-structured NIST DB
                for fine_entry in self.nist_fine_db:
                    if fine_entry['shell_lower'] == lower_shell and fine_entry['shell_upper'] == upper_shell:
                        f_fine = fine_entry['f']
                        if f_fine <= 0:
                            continue

                        # Fractionate the averaged macro alpha using the true fine-structured f ratio
                        fraction = f_fine / sum_nist_f
                        alpha_fine = alpha_avg * fraction

                        # Retrieve the Vlcek mapping for these exact fine-structured states
                        vlcek_m = self._map_nist_to_vlcek(fine_entry['qns_lower'])
                        vlcek_n = self._map_nist_to_vlcek(fine_entry['qns_upper'])

                        if not vlcek_m or not vlcek_n:
                            continue

                        m, n = (vlcek_m, vlcek_n) if self.levels_db[vlcek_m]['excitation_energy'] < \
                                                     self.levels_db[vlcek_n]['excitation_energy'] else (vlcek_n,
                                                                                                        vlcek_m)

                        if self.levels_db[m]['excitation_energy'] >= self.levels_db[n]['excitation_energy']:
                            continue

                        # Vlcek upper limit boundary for Kimura parameters
                        if n > 45:
                            continue

                        trans_key = (m, n)

                        if trans_key not in self.allowed_kimura_db:
                            exc_energy = self.levels_db[n]['excitation_energy'] - self.levels_db[m]['excitation_energy']
                            self.allowed_kimura_db[trans_key] = {'excitation_energy': exc_energy, 'alpha': 0.0,
                                                                 'beta': beta_val}

                        # Accumulate alphas safely. Multiple fine lines resolving to the same
                        # lumped Vlcek transition will correctly sum their assigned fractions here.
                        self.allowed_kimura_db[trans_key]['alpha'] += alpha_fine

            else:
                # Path B: Unsupervised Statistical Redistribution
                for core, a_mac in alpha_macro.items():
                    lower_matches = self._match_kimura_and_vlcek(lower_shell, core)
                    upper_matches = self._match_kimura_and_vlcek(upper_shell, core)

                    if not lower_matches or not upper_matches:
                        continue

                    core_g = 4 if core == "3/2" else 2
                    g_upper_kimura = core_g * 2 * (2 * l_upper_val + 1)

                    if g_upper_kimura <= 0:
                        raise ValueError(
                            f"CRITICAL: Theoretical g_upper evaluated to zero or less for shell {upper_shell}.")

                    for m in lower_matches:
                        for n in upper_matches:
                            vlcek_lower = min(m, n, key=lambda x: self.levels_db[x]['excitation_energy'])
                            vlcek_upper = max(m, n, key=lambda x: self.levels_db[x]['excitation_energy'])

                            if self.levels_db[vlcek_lower]['excitation_energy'] >= self.levels_db[vlcek_upper][
                                'excitation_energy']:
                                continue

                            # Vlcek upper limit boundary
                            if vlcek_upper > 45:
                                continue

                            # Distribute proportionally based strictly on the target upper state
                            alpha_mn = self.distribute_multiplet(a_mac, vlcek_upper, core, l_upper_val)

                            trans_key = (vlcek_lower, vlcek_upper)

                            if trans_key not in self.allowed_kimura_db:
                                exc_energy = self.levels_db[vlcek_upper]['excitation_energy'] - \
                                             self.levels_db[vlcek_lower]['excitation_energy']
                                self.allowed_kimura_db[trans_key] = {'excitation_energy': exc_energy, 'alpha': 0.0,
                                                                     'beta': beta_val}

                            self.allowed_kimura_db[trans_key]['alpha'] += alpha_mn

        # Hydrogenic algorithmic ladder (45 <= m < n <= 65)
        for trans_key, data in self.einstein_hydrogenic_db.items():
            if trans_key not in self.allowed_kimura_db:
                lower_lvl, upper_lvl = trans_key
                exc_energy = self.levels_db[upper_lvl]['excitation_energy'] - self.levels_db[lower_lvl][
                    'excitation_energy']

                # For hydrogenic approximation, K is assumed 1.0 (alpha = 1.0 * f_mn)
                self.allowed_kimura_db[trans_key] = {
                    'excitation_energy': exc_energy,
                    'alpha': 1.0 * data['f'],
                    'beta': 1.0
                }

    def process_parity_forbidden_data(self):
        print("Processing Kimura Parity-Forbidden transitions...")
        for row in self.raw_data['kimura_parameters']:
            lower_shell, upper_shell = row['lower_shell'].strip(), row['upper_shell'].strip()

            l_lower_val = "spdfghiklmno".find(lower_shell[-1].lower())
            l_upper_val = "spdfghiklmno".find(upper_shell[-1].lower())

            # Filter for Parity-Forbidden (|Delta l| != 1)
            if abs(l_lower_val - l_upper_val) == 1: continue

            for core, col_name in [("3/2", 'alpha_2P3_2'), ("1/2", 'alpha_2P1_2')]:
                val_raw = row.get(col_name, '').strip()
                if not val_raw:
                    continue
                alpha_macro = float(val_raw)

                # Calculate theoretical capacities to prevent lumped state fraction explosion
                core_g = 4 if core == "3/2" else 2
                g_lower_kimura = core_g * 2 * (2 * l_lower_val + 1)
                g_upper_kimura = core_g * 2 * (2 * l_upper_val + 1)

                lower_matches = self._match_kimura_and_vlcek(lower_shell, core)
                upper_matches = self._match_kimura_and_vlcek(upper_shell, core)
                if not lower_matches or not upper_matches:
                    continue

                for m in lower_matches:
                    for n in upper_matches:

                        # Check Energy and Vlcek's Hard Cap (n <= 47)
                        if self.levels_db[m]['excitation_energy'] >= self.levels_db[n]['excitation_energy']: continue
                        if n > 47: continue

                        # Distribute proportionally based strictly on the target upper state
                        split_alpha = self.distribute_multiplet(alpha_macro, n, core, l_upper_val)

                        trans_key = (m, n)

                        if trans_key not in self.parity_forbidden_db:
                            exc_energy = self.levels_db[n]['excitation_energy'] - self.levels_db[m]['excitation_energy']
                            self.parity_forbidden_db[trans_key] = {'excitation_energy': exc_energy, 'alpha': 0.0}
                        self.parity_forbidden_db[trans_key]['alpha'] += split_alpha

    def process_spin_forbidden_data(self):
        print("Processing Spin-Forbidden transitions...")
        for row in self.raw_data['vlcek_spin_forb']:
            m, n = int(row['lower_n']), int(row['upper_n'])
            trans_key = (m, n)
            exc_energy = self.levels_db[n]['excitation_energy'] - self.levels_db[m]['excitation_energy']
            self.spin_forbidden_db[trans_key] = {'excitation_energy': exc_energy, 'alpha': float(row['alpha']),
                                                 'equation': int(row['equation'])}

        # --- Diagnostic Trackers ---
        added_spin_forbidden = 0
        skipped_existing = 0

        for m in range(2, 48):
            for n in range(m + 1, 48):
                # Enforce strict energy ordering: lower_lvl must have the smaller excitation energy
                if self.levels_db[m]['excitation_energy'] < self.levels_db[n]['excitation_energy']:
                    lower_lvl, upper_lvl = m, n
                else:
                    lower_lvl, upper_lvl = n, m

                trans_key = (lower_lvl, upper_lvl)

                # Skip transitions that were already populated manually or by prior specialized logic
                if trans_key in self.spin_forbidden_db:
                    skipped_existing += 1
                    continue

                # Evaluate quantum numbers to determine if an electron-exchange (spin-forbidden) collision is required
                is_spin_forbidden = False
                for qn_lower in self.levels_db[lower_lvl]['quantum_numbers']:
                    for qn_upper in self.levels_db[upper_lvl]['quantum_numbers']:
                        if self._categorize_transition(qn_lower, qn_upper) == 'S':
                            is_spin_forbidden = True
                            break
                    if is_spin_forbidden:
                        break

                # Register the transition as an algorithmic fallback for the electron-exchange process
                if is_spin_forbidden:
                    exc_energy = self.levels_db[upper_lvl]['excitation_energy'] - self.levels_db[lower_lvl][
                        'excitation_energy']
                    self.spin_forbidden_db[trans_key] = {
                        'excitation_energy': exc_energy,
                        'alpha': 0.1,
                        'equation': 0
                    }
                    added_spin_forbidden += 1

        # --- Diagnostic Console Report ---
        print("\n  --- SPIN-FORBIDDEN FALLBACK REPORT ---")
        print(f"  [+] Added {added_spin_forbidden} new spin-forbidden transitions.")
        print(f"  [-] Skipped {skipped_existing} transitions (Already populated in Spin-Forbidden DB).")
        print("  --------------------------------------\n")

    def process_photoionization_data(self):
        """
        Calculates effective gamma weights for photoionization based on Katsonis (1976).
        Aggregates exact statistical weights from parsed J values to accurately calculate
        the mu fill-fractions for partially and fully lumped configurations.
        """
        print("Processing Photoionization gammas...")

        katsonis_dict = {
            (int(row['n']), int(row['l'])): float(row['gamma'])
            for row in self.raw_data['katsonis_photo']
        }

        vlcek_dict = {
            int(row['n']): float(row['gamma'])
            for row in self.raw_data['vlcek_photo']
        }

        for m, data in self.levels_db.items():
            designation = data.get('designation', 'Unknown')

            # Boundary constraints
            if m == 1:
                continue
            if m > 45:
                self.photoionization_db[m] = 1.0
                continue

            # 4s explicit overrides
            if 2 <= m <= 5:
                gamma_val = vlcek_dict.get(m)
                if gamma_val is not None:
                    self.photoionization_db[m] = gamma_val
                else:
                    print(f"[WARN] Level {m} ({designation}): Missing Vlcek gamma. Defaulting to 0.0.")
                    self.photoionization_db[m] = 0.0
                continue

            # Intermediate state processing via bottom-up J-aggregation
            qns = data.get('quantum_numbers', [])
            config_g_accumulator = {}
            skip_level = False

            for qn in qns:
                if 'n' not in qn or 'l' not in qn:
                    continue

                n_raw, l_raw, core, j_raw = qn['n'], qn['l'], qn.get('core'), qn.get('J', 'all')

                try:
                    n_val = int(n_raw)
                except (ValueError, TypeError):
                    continue

                if n_val > 9:
                    print(f"[INFO] Level {m} ({designation}): Contains n={n_val} > 9. Defaulting to Kramers.")
                    skip_level = True
                    break

                # Helper to calculate theoretical capacity based on the core
                def get_capacity(l_electron, j_core):
                    if j_core == '1/2':
                        return 4.0 * (2 * l_electron + 1)
                    elif j_core == '3/2':
                        return 8.0 * (2 * l_electron + 1)
                    else:
                        raise ValueError(f"Invalid core value: {j_core}")

                # Handle fully lumped principal shells (l = 'all')
                if str(l_raw).lower() == 'all':
                    for l_val in range(n_val):
                        cap = get_capacity(l_val, core)
                        config_key = (n_val, l_val, core)
                        # For a fully lumped shell, it occupies 100% of its capacity
                        config_g_accumulator[config_key] = config_g_accumulator.get(config_key, 0) + cap
                    continue

                # Handle standard/partial configurations
                try:
                    l_val = int(l_raw)
                except (ValueError, TypeError):
                    continue

                cap = get_capacity(l_val, core)
                config_key = (n_val, l_val, core)

                # Determine the actual statistical weight contribution of this specific state
                if str(j_raw).lower() == 'all':
                    g_state = cap
                else:
                    try:
                        # 2J + 1 logic for resolved fine-structure states
                        g_state = 2.0 * float(j_raw) + 1.0
                    except (ValueError, TypeError):
                        g_state = cap

                config_g_accumulator[config_key] = config_g_accumulator.get(config_key, 0) + g_state

            if skip_level:
                continue

            if not config_g_accumulator:
                print(f"[WARN] Level {m} ({designation}): No valid configurations parsed. Defaulting to 0.0.")
                self.photoionization_db[m] = 0.0
                continue

            gamma_p = 0.0

            # Calculate the final Katsonis weighted sum
            for (n, l, core), actual_g in config_g_accumulator.items():
                gamma = katsonis_dict.get((n, l))
                if gamma is None:
                    print(f"[WARN] Level {m} ({designation}): Missing Table 3 gamma for n={n}, l={l}.")
                    continue

                theoretical_capacity = get_capacity(l, core)
                mu_nl = actual_g / theoretical_capacity

                gamma_p += gamma * mu_nl

            self.photoionization_db[m] = gamma_p

    def process_metastable_quenching_data(self):
        print("Processing Metastable Quenching parameters...")
        t_ref, bessel_zero = 300.0, 2.405

        for row in self.raw_data['tachibana']:
            n_level = int(row['n'])
            D_0, K_2 = float(row['D_0']), float(row['K_2'])
            self.metastable_quenching_db[n_level] = {
                'd_n': D_0 * (bessel_zero ** 2) / (t_ref ** 0.73),
                'B_n': K_2
            }


class CRDataExporter:
    """
    Load Phase: Takes the processed dictionaries and outputs them to finalized CSVs.
    """

    def __init__(self, script_dir, processor):
        self.output_dir = Path(script_dir) / 'processed'

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"  [!] CRITICAL ERROR: Could not create output directory '{self.output_dir}': {e}")

        self.processor = processor

    def export_all(self):
        print("=== Exporting CR Matrices ===")
        self.export_levels_csv()
        self.export_einstein_csv()
        self.export_allowed_csv()

        # Note: Assuming the remaining forbidden/ionization methods are structurally unchanged
        self.export_parity_forbidden_csv()
        self.export_spin_forbidden_csv()
        self.export_ionization_csv()
        self.export_photoionization_csv()
        self.export_metastable_quenching_csv()
        print("=== All Exports Complete ===")

    def export_levels_csv(self):
        """
        Exports the mapped and sorted atomic levels along with their lumped quantum numbers.
        """
        output_file = self.output_dir / 'excitation_levels.csv'
        print(f"Exporting processed levels to {output_file}...")

        output_rows = []
        for n_level, data in self.processor.levels_db.items():
            n_list, l_list, k_list, j_list, core_list = [], [], [], [], []
            for q in data['quantum_numbers']:
                n_list.append(str(q['n']))
                l_list.append(str(q['l']))
                k_list.append(str(q['K']))
                j_list.append(str(q['J']))
                core_list.append(str(q['core']))

            qn_K_str = "all" if all(k == "all" for k in k_list) else " / ".join(k_list)
            qn_J_str = "all" if all(j == "all" for j in j_list) else " / ".join(j_list)

            output_rows.append({
                'n': n_level, 'designation': data['designation'],
                'excitation_energy': f"{data['excitation_energy']:.4f}", 'g': data['g'],
                'qn_n': " / ".join(n_list), 'qn_l': " / ".join(l_list),
                'qn_K': qn_K_str, 'qn_J': qn_J_str, 'qn_core': " / ".join(core_list)
            })

        try:
            with open(output_file, mode='w', encoding='utf-8', newline='') as f:
                fieldnames = ['n', 'designation', 'excitation_energy', 'g', 'qn_n', 'qn_l', 'qn_K', 'qn_J', 'qn_core']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)
        except IOError as e:
            print(f"  [!] CRITICAL ERROR: Failed to write to {output_file}: {e}")

    def export_einstein_csv(self):
        """
        Exports unified Einstein A and f_ik parameters.
        Pulls from the pre-averaged NIST database, the statistically distributed
        Kimura database, and the algorithmic Hydrogenic database.
        """
        output_file = self.output_dir / 'einstein_parameters.csv'
        print(f"Exporting processed Einstein parameters to {output_file}...")
        output_rows = []

        # Add NIST transitions
        for (lower_n, upper_n), data in self.processor.einstein_nist_db.items():
            output_rows.append({
                'upper_n': upper_n,
                'lower_n': lower_n,
                'A': f"{data['A']:.6e}",
                'f': f"{data['f']:.6e}",
                'source': 'NIST'
            })

        # Add Kimura transitions
        for (lower_n, upper_n), data in self.processor.einstein_kimura_db.items():
            output_rows.append({
                'upper_n': upper_n,
                'lower_n': lower_n,
                'A': f"{data['A']:.6e}",
                'f': f"{data['f']:.6e}",
                'source': 'Kimura'
            })

        # Add Hydrogenic transitions
        for (lower_n, upper_n), data in self.processor.einstein_hydrogenic_db.items():
            output_rows.append({
                'upper_n': upper_n,
                'lower_n': lower_n,
                'A': f"{data['A']:.6e}",
                'f': f"{data['f']:.6e}",
                'source': 'Hydrogenic'
            })

        # Standardize sorting to ensure predictable structure
        output_rows.sort(key=lambda x: (x['lower_n'], x['upper_n']))

        try:
            with open(output_file, mode='w', encoding='utf-8', newline='') as f:
                fieldnames = ['upper_n', 'lower_n', 'A', 'f', 'source']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)
        except IOError as e:
            print(f"  [!] CRITICAL ERROR: Failed to write to {output_file}: {e}")

    def export_allowed_csv(self):
        """
        Exports Kimura cross-section parameters for dipole-allowed transitions.
        """
        output_file = self.output_dir / 'excitation_allowed.csv'
        print(f"Exporting Allowed transitions to {output_file}...")

        # Unify ground and inter-state dictionaries safely
        combined_allowed_db = {**self.processor.allowed_ground_db, **self.processor.allowed_kimura_db}

        output_rows = []
        for (lower_n, upper_n) in sorted(combined_allowed_db.keys()):
            data = combined_allowed_db[(lower_n, upper_n)]
            output_rows.append({
                'lower_n': lower_n, 'upper_n': upper_n,
                'excitation_energy': f"{data['excitation_energy']:.4f}",
                'alpha': f"{data['alpha']:.6e}", 'beta': f"{data['beta']}"
            })

        try:
            with open(output_file, mode='w', encoding='utf-8', newline='') as f:
                fieldnames = ['lower_n', 'upper_n', 'excitation_energy', 'alpha', 'beta']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)
        except IOError as e:
            print(f"  [!] CRITICAL ERROR: Failed to write to {output_file}: {e}")

    def export_parity_forbidden_csv(self):
        output_file = self.output_dir / 'excitation_parity_forbidden.csv'
        print(f"Exporting Parity-Forbidden transitions to {output_file}...")
        output_rows = []

        for (lower_n, upper_n) in sorted(self.processor.parity_forbidden_db.keys()):
            data = self.processor.parity_forbidden_db[(lower_n, upper_n)]
            output_rows.append({
                'lower_n': lower_n, 'upper_n': upper_n,
                'excitation_energy': f"{data['excitation_energy']:.4f}",
                'alpha': f"{data['alpha']:.6e}"
            })

        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['lower_n', 'upper_n', 'excitation_energy', 'alpha'])
            writer.writeheader()
            writer.writerows(output_rows)

    def export_spin_forbidden_csv(self):
        output_file = self.output_dir / 'excitation_spin_forbidden.csv'
        print(f"Exporting Spin-Forbidden transitions to {output_file}...")
        output_rows = []

        for (lower_n, upper_n) in sorted(self.processor.spin_forbidden_db.keys()):
            data = self.processor.spin_forbidden_db[(lower_n, upper_n)]
            output_rows.append({
                'lower_n': lower_n, 'upper_n': upper_n,
                'excitation_energy': f"{data['excitation_energy']:.4f}",
                'alpha': f"{data['alpha']:.6e}", 'equation': data['equation']
            })

        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['lower_n', 'upper_n', 'excitation_energy', 'alpha', 'equation'])
            writer.writeheader()
            writer.writerows(output_rows)

    def export_ionization_csv(self):
        output_file = self.output_dir / 'ionization.csv'
        print(f"Exporting Ionization transitions to {output_file}...")
        output_rows = []

        for n_level in sorted(self.processor.ionization_db.keys()):
            data = self.processor.ionization_db[n_level]
            output_rows.append({
                'level_n': n_level, 'ionization_energy': f"{data['ionization_energy']:.7f}",
                'xi_ion': data['xi_ion'], 'alpha_ion': f"{data['alpha_ion']:.2f}", 'beta_ion': f"{data['beta_ion']}"
            })

        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['level_n', 'ionization_energy', 'xi_ion', 'alpha_ion', 'beta_ion'])
            writer.writeheader()
            writer.writerows(output_rows)

    def export_photoionization_csv(self):
        output_file = self.output_dir / 'photoionization.csv'
        print(f"Exporting Photoionization transitions to {output_file}...")
        output_rows = [{'n': n_level, 'gamma_p': f"{gamma:.6e}"} for n_level, gamma in
                       sorted(self.processor.photoionization_db.items())]

        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['n', 'gamma_p'])
            writer.writeheader()
            writer.writerows(output_rows)

    def export_metastable_quenching_csv(self):
        output_file = self.output_dir / 'metastable_quenching.csv'
        print(f"Exporting Metastable Quenching parameters to {output_file}...")
        output_rows = []

        for n_level in sorted(self.processor.metastable_quenching_db.keys()):
            data = self.processor.metastable_quenching_db[n_level]
            output_rows.append({'level_n': n_level, 'd_n': f"{data['d_n']:.4e}", 'B_n': f"{data['B_n']:.2e}"})

        with open(output_file, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['level_n', 'd_n', 'B_n'])
            writer.writeheader()
            writer.writerows(output_rows)


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    current_script_dir = Path(__file__).resolve().parent

    # 1. Extract
    loader = CRDataLoader(script_dir=current_script_dir)
    raw_csv_data = loader.load_all_raw()

    # 2. Transform
    processor = CRDataProcessor(raw_data=raw_csv_data)
    processor.process_all()

    # 3. Load / Export
    exporter = CRDataExporter(script_dir=current_script_dir, processor=processor)
    exporter.export_all()