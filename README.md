# crm-argon: Argon Collisional-Radiative Model

**crm-argon** is a comprehensive Python package for modeling argon plasma kinetics using a Collisional-Radiative Model (CRM) approach. 

The primary objective of this package is to serve as a diagnostic tool for experimental plasma physics. By taking relative experimental population densities (inferred externally from Optical Emission Spectroscopy line intensities), the included optimizer varies the electron temperature ($T_e$) and/or the electron density ($n_e$) to find the plasma parameters that best fit the experimental data.

## 🚀 Quick Start

You can run the model directly from the command line using the built-in package execution. All physics parameters, modes, and plotting options are controlled via TOML files.

**1. Run a Standard Simulation Sweep**
To run forward calculations (predicting state populations based on given plasma parameters):
```bash
python -m crm_argon --model configs/model_config.toml --run configs/run_simulation.toml
```

**2. Run a Parameter Optimization**
To fit plasma parameters ($T_e$ or $n_e$) to your experimental data:
```bash
python -m crm_argon --model configs/model_config.toml --run configs/run_optimization.toml
```

*Note: Check the provided files in the `configs/` directory to see examples of how to set your parameters, toggle export settings, and format your experimental data!*

---

## 📖 Physics Overview

This CRM simulates the atomic state of an argon gas by calculating all possible transitions between different states of the electron shell. The electron shell is divided into 65 lumped sublevels of approximately equal excitation energies. 

### Respected Processes
The following processes are considered in the CRM:

1.  **Electron Impact:** Excitation and de-excitation.
2.  **Electron Impact Ionization:** Including three-body recombination (where the third body is an electron).
3.  **Atom Impact:** Excitation and de-excitation.
4.  **Atom Impact Ionization:** Including three-body recombination (where the third body is an atom).
5.  **Radiative Processes:** Spontaneous emission and photoexcitation (bound-bound).
6.  **Recombination:** Radiative recombination and photoionization (bound-free).
7.  **Metastable Quenching:** Collisional de-excitation and diffusion specifically for the 4s metastable states.

### Rate Equation
The time evolution for the number density of level $n_{m}$ is governed by the following population rate equation:

$$
\begin{aligned}
\frac{\mathrm{d}n_m}{\mathrm{d}t} = & + n_n \Bigg[ \sum_{n<m} (C_{mn}n_e + K_{mn}n_1) + \sum_{n>m} (F_{mn}n_e + L_{mn}n_1 + \Lambda_{mn}A_{mn}) \Bigg] \\
& + n_e n_+ (O_m n_e + \Lambda_m R_m + W_m n_1) \\
& - n_m \Bigg[ \sum_{n>m} (C_{nm}n_e + K_{nm}n_1) + \sum_{n<m} (F_{nm}n_e + L_{nm}n_1 + \Lambda_{nm}A_{nm}) \\
& \qquad\quad  + S_m n_e + V_m n_1 + \frac{D_m}{\Lambda^2} + n_1^2 B_m \Bigg],
\end{aligned}
$$

Assuming a quasi-stationary state ($dn/dt=0$), this simplifies to a set of coupled linear equations.

---

## 📚 References & Data Sources

This implementation relies on the following foundational research and atomic databases:

### Methodology
* **Vlcek, J. (1989)**. "A collisional-radiative model applicable to argon discharges over a wide range of conditions. I. Formulation and basic data", *Journal of Physics D: Applied Physics* 22, 623–631.
* **Bogaerts, A., Gijbels, R., and Vlcek, J. (1998)**. "Collisional-radiative model for an argon glow discharge", *Journal of Applied Physics* 84, 121–136.
* **Akatsuka, H. (2009)**. "Excited level populations and excitation kinetics of nonequilibrium ionizing argon discharge plasma of atmospheric pressure", *Physics of Plasmas* 16, 043502.

### Atomic Data
* **Kramida, A., Ralchenko, Y., and Reader, J. (2024)**. *NIST Atomic Spectra Database*, version 5.12 (National Institute of Standards and Technology).
* **Kimura, A., Kobayashi, H., Nishida, M., and Valentin, P. (1985)**. "Calculation of collisional and radiative transition probabilities between excited argon levels", *Journal of Quantitative Spectroscopy and Radiative Transfer* 34, 189–215.
* **Katsonis, K. (1976)**. "Statistical and Kinetic Study of Ar non-LTE Plasmas", PhD thesis (Paris-Sud University, Paris), 53–83.
* **Tachibana, K. (1986)**. "Excitation of the $1s_5$, $1s_4$, $1s_3$, and $1s_2$ levels of argon by low-energy electrons", *Physical Review A* 34, 1007–1015.
* **Baranov, I., Demidov, V., and Kolokolov, N. (1981)**. "Temperature dependence of rate constants for metastable atomic-argon deactivation by slow electrons", *Optical Spectroscopy* 51, 571–574.

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**. 

This ensures that the code remains open source and that any further developments or derivatives of this model by the community are shared under the same freedom. See the [LICENSE](LICENSE) file for the full legal text.

## ✉️ Contact & Documentation

For a detailed explanation of the physics, the underlying methodology, and the verification of this model, please refer to the comprehensive project report available at: 

**[https://peterplr.github.io/](https://peterplr.github.io/)**

For questions, feedback, or collaboration inquiries, please use the contact form provided on the website.