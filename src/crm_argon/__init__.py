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

"""
Argon Collisional-Radiative (CR) Model Package.

This package provides a comprehensive framework for modeling argon plasma kinetics
by using a collisional-radiative approach. It includes a database of atomic data,
cross-sectional data, and a physics engine for calculating transition rates.

Main Components:
----------------
- Interface: The primary entry point for high-level simulation control.
- PhysicsEngine: The core engine for calculating transition rates and matrix assembly.
- Kinetics: Management of physical processes and rate coefficient calculations.
- CrossSections: Calculation of cross-sectional data from database.
- Database: Internal handler for level properties and cross-section data.

Usage:
------
    >>> from cr_model import Interface
    >>> model = Interface(config_file="config.json")
    >>> results = model.run(Te_eV=3.0, Tg_K=300, n_e_cm3=1e11, n_1_cm3=1e15, n_ion_cm3=1e11, R_cm=1.0)
"""

# Import core classes to the package level for easier access
from .interface import Interface
from .physicsEngine import PhysicsEngine
from .kinetics import Kinetics
from .database import Database
from .eedf import EEDF
from .solver import SteadyStateSolver
from .spectrum import SyntheticOES
from .plotter import Plotter
from .exporter import Exporter
from .diagnostics import FluxAnalyzer

# Expose constants selectively or via sub-module
from . import constants

__all__ = [
    "Interface",
    "PhysicsEngine",
    "Kinetics",
    "Database",
    "EEDF",
    "SteadyStateSolver",
    "SyntheticOES",
    "Plotter",
    "Exporter",
    "FluxAnalyzer",
    "constants"
]

__version__ = "1.0.0"
__author__ = "Peter Preisler"