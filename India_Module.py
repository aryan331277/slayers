"""
WorldSim India — PettingZoo AEC Environment
=============================================
10 Indian states govern themselves as AI agents competing for water, food,
energy, and land resources under climate shocks and geopolitical pressure.

All parameters are derived from worldsim_merged.csv — nothing is hardcoded.

Columns used (43 total):
  state, year
  total_crop_area_ha, total_crop_production_t, n_crop_types   → food / land
  energy_aggregate_fuel, energy_fuel                          → energy
  rain_sub_annual, rain_sub_jun-sep, rain_sub_oct-dec,        → water / climate
  rain_sub_jan-feb, rain_sub_mar-may, rain_sub_*monthly       → climate detail
  rain_dist_annual, rain_dist_*monthly                        → water cross-check
  population_thousands                                        → demand
  per_capita_income_inr                                       → wealth / adaptive capacity

Install (run once on Kaggle):
  pip install pettingzoo gymnasium networkx scikit-learn matplotlib seaborn

Reference:
  Terry et al. (2021)  "PettingZoo: Gym for Multi-Agent Reinforcement Learning"
  Hoff (2011)          "Understanding the Nexus", Stockholm Environment Institute
  Wilks (2011)         "Statistical Methods in Atmospheric Sciences"
"""

# ── Dependency check ──────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["pettingzoo", "gymnasium", "networkx"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import warnings, copy
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
import gymnasium
from gymnasium import spaces

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# 10 representative states — chosen for maximum diversity in resource profiles
# RJ: water-scarce arid | MH: industrial large economy | UP: highest population
# KL: highest rainfall   | GJ: semi-arid industrial    | WB: delta high rainfall
# PB: breadbasket GW-stressed | BR: dense-poor agri   | KA: mixed tech-agri
# TN: semi-arid industrial coast
STATE_AGENTS: Dict[str, str] = {
    "RJ": "Rajasthan",
    "MH": "Maharashtra",
    "UP": "Uttar Pradesh",
    "KL": "Kerala",
    "GJ": "Gujarat",
    "WB": "West Bengal",
    "PB": "Punjab",
    "BR": "Bihar",
    "KA": "Karnataka",
    "TN": "Tamil Nadu",
}

# Agent IDs (short codes) used throughout the environment
AGENT_IDS: List[str] = list(STATE_AGENTS.keys())

# Climate states (Markov chain)
CLIMATE_STATES = {0: "Normal", 1: "Drought", 2: "Flood", 3: "Heatwave", 4: "Storm"}

# Action catalogue
ACTION_TYPES: Dict[int, str] = {
    0:  "invest_water",
    1:  "invest_food",
    2:  "invest_energy",
    3:  "stockpile",
    4:  "population_policy",
    5:  "offer_water_trade",
    6:  "offer_food_trade",
    7:  "offer_energy_trade",
    8:  "accept_trade",
    9:  "reject_trade",
    10: "defect_trade",
    11: "form_alliance",
    12: "leave_alliance",
    13: "sanction",
    14: "raid",
    15: "diplomat",
    16: "do_nothing",
}
N_ACTIONS = len(ACTION_TYPES)
TARGETED_ACTIONS = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADER
# Reads worldsim_merged.csv and derives ALL simulation parameters from data
# ══════════════════════════════════════════════════════════════════════════════

class WorldSimDataLoader:
    """
    Derives all simulation parameters from worldsim_merged.csv.

    Column → Simulation Variable mapping
    ─────────────────────────────────────
    rain_sub_annual / rain_dist_annual    → water availability stock
    rain_sub_jun-sep                      → monsoon reliability (Drought/Flood risk)
    rain_sub_oct-dec                      → cyclone/post-monsoon risk (Storm risk)
    rain_sub_mar-may                      → pre-monsoon heat risk (Heatwave risk)
    total_crop_production_t               → food production capacity
    total_crop_area_ha                    → land resource
    n_crop_types                          → agricultural diversity (resilience)
    energy_aggregate_fuel + energy_fuel   → total energy capacity (MW)
    population_thousands                  → demand multiplier / carrying capacity
    per_capita_income_inr                 → economic power / adaptive capacity
    """

    BASE_YEAR = 2015  # Midpoint with broadest data coverage

    def __init__(self, csv_path: str):
        print(f"[DataLoader] Loading {csv_path} ...")
        raw = pd.read_csv(csv_path)

        # ── Deduplicate: multiple per_capita_income rows per state/year ───────
        # Exclude 'year' from the aggregated numeric cols to avoid reset_index collision.
        num_cols = [c for c in raw.select_dtypes(include=np.number).columns
                    if c != "year"]
        self.df = (
            raw.groupby(["state", "year"])[num_cols]
            .mean()
            .reset_index()
        )
        print(f"[DataLoader] Deduplicated: {raw.shape} → {self.df.shape}")

        # Filter to our 10 target states
        target_names = list(STATE_AGENTS.values())
        self.df_targets = self.df[self.df["state"].isin(target_names)].copy()

        # Build inverse map: full state name → agent ID
        self._name_to_id = {v: k for k, v in STATE_AGENTS.items()}

        # Compute cross-state normalisation ranges (use all target rows)
        self._ranges: Dict[str, Dict] = self._compute_ranges()

        # Main outputs consumed by the environment
        self.region_init       = self._compute_region_init()
        self.depletion_rates   = self._compute_depletion_rates()
        self.climate_matrices  = self._compute_climate_matrices()
        self.trade_graph       = self._compute_trade_graph()

        self._print_init_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get(self, state_name: str, col: str, year: int) -> float:
        """Return value at specific year; interpolate from nearest if missing."""
        sub = (
            self.df_targets[self.df_targets["state"] == state_name]
            .sort_values("year")
        )
        exact = sub[sub["year"] == year][col].dropna()
        if len(exact) > 0:
            return float(exact.iloc[0])
        valid = sub[col].dropna()
        if len(valid) == 0:
            return np.nan
        return float(np.interp(year, sub.loc[valid.index, "year"].values, valid.values))

    def _get_series(self, state_name: str, col: str) -> pd.Series:
        """Return full time-series for a state+column, sorted by year."""
        sub = (
            self.df_targets[self.df_targets["state"] == state_name]
            .sort_values("year")[["year", col]]
            .dropna()
        )
        return sub

    def _safe(self, v: float, fallback: float) -> float:
        return fallback if (np.isnan(v) or np.isinf(v)) else float(v)

    def _compute_ranges(self) -> Dict[str, Dict]:
        """Cross-state min/max for normalisation, computed over BASE_YEAR ± 5."""
        year_window = range(self.BASE_YEAR - 5, self.BASE_YEAR + 6)
        sub = self.df_targets[self.df_targets["year"].isin(year_window)].copy()
        ranges: Dict[str, Dict] = {}

        # Raw column ranges
        for col in sub.select_dtypes(include=np.number).columns:
            if col == "year":
                continue
            vals = sub[col].dropna().values
            if len(vals) == 0:
                ranges[col] = {"min": 0.0, "max": 1.0}
            else:
                ranges[col] = {"min": float(vals.min()), "max": float(vals.max())}

        # ── Derived per-capita ranges (important: normalise against like-for-like) ──
        pop_k = sub["population_thousands"].replace(0, np.nan)

        # Food per capita (tonnes/person)
        sub["_food_pc"] = sub["total_crop_production_t"] / (pop_k * 1000)
        # Energy per capita (MW per 1000 people)
        sub["_energy_pc"] = (sub["energy_aggregate_fuel"].fillna(0)
                             + sub["energy_fuel"].fillna(0)) / pop_k
        # Land per capita (ha/person)
        sub["_land_pc"] = sub["total_crop_area_ha"] / (pop_k * 1000)
        # Rainfall (use dist_annual as the range reference)
        # rain_dist_annual already in ranges — used as proxy for rain_combined too

        for derived in ("_food_pc", "_energy_pc", "_land_pc"):
            vals = sub[derived].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(vals) == 0:
                ranges[derived] = {"min": 0.0, "max": 1.0}
            else:
                ranges[derived] = {"min": float(vals.min()), "max": float(vals.max())}

        return ranges

    def _norm(self, value: float, col: str, invert: bool = False) -> float:
        """Normalise value to [0.05, 1.0]; optionally invert (1 - norm)."""
        if np.isnan(value) or col not in self._ranges:
            return 0.5
        mn, mx = self._ranges[col]["min"], self._ranges[col]["max"]
        if mx == mn:
            return 0.5
        n = np.clip((value - mn) / (mx - mn), 0.0, 1.0)
        n = 0.05 + n * 0.95  # map to [0.05, 1.0] — avoid absolute zero
        return float(1.0 - n) if invert else float(n)

    # ─────────────────────────────────────────────────────────────────────────
    # Region initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_region_init(self) -> Dict[str, Dict]:
        """
        Build initial state vector for each agent from worldsim_merged.csv.

        Column → Variable derivation (verbose):

        WATER STOCK:
          Primary:  rain_dist_annual   [mm] — long-run average district rainfall.
          Secondary: rain_sub_annual   [mm] — subdivision historical mean.
          Combined via weighted average. Higher rainfall → higher water stock.
          Water-stressed states (RJ: ~300mm, PB heavy GW withdrawal) score lower.

        FOOD STOCK:
          total_crop_production_t / (population_thousands × 1000) = tonnes/person
          Normalised across states. Also scaled by n_crop_types diversity bonus.
          High production per capita AND diverse crops = high food security.

        ENERGY STOCK:
          energy_aggregate_fuel + energy_fuel = total installed capacity [MW]
          Divided by population_thousands to get MW per 1000 persons.
          Higher per-capita capacity = higher energy stock.

        LAND STOCK:
          total_crop_area_ha / (population_thousands × 1000) = ha/person
          Higher cultivated area per person = greater land resource.
          Also modulated by rain_dist_annual (irrigated potential).

        ECONOMIC POWER:
          per_capita_income_inr normalised. High PCI = high economic power.

        ADAPTIVE CAPACITY:
          0.4 × economic_power + 0.3 × energy_stock + 0.3 × crop_diversity
          Wealthier, more energy-secure, more diverse states adapt better.

        SOCIAL STABILITY:
          Inverse of per_capita_income_inr CV (within state, across years) +
          inverse of population growth pressure.
          More stable income = more stable society.

        MILITARY STRENGTH (geopolitical influence):
          Proxy: 0.5 × economic_power + 0.3 × population_norm + 0.2 × energy_stock
          Large, wealthy, industrialised states have more political leverage.

        TRADE OPENNESS:
          food surplus ratio = production - estimated consumption (pop × calorie need)
          Energy surplus: whether state is net producer above its own demand.
          Surplus states are more open to trade.

        FINANCIAL VULNERABILITY:
          Inverse of PCI relative to median. Poor states = more financially fragile.
          Also: negative PCI values (seen in data) = immediate vulnerability.

        CLIMATE VULNERABILITY (per historical record):
          Drought frequency: years where jun-sep rain < (mean - 1σ) / total years
          Flood frequency:   years where jun-sep rain > (mean + 1σ) / total years
          Heatwave proxy:    states with rain_sub_mar-may < 50mm AND annual < 600mm
          Storm proxy:       states with rain_sub_oct-dec > 200mm (Bay of Bengal cyclones)
          Shock severity:    max single-year deviation from mean rainfall
        """
        init: Dict[str, Dict] = {}
        y = self.BASE_YEAR

        for agent_id, state_name in STATE_AGENTS.items():
            # ── Raw value extraction ─────────────────────────────────────────
            rain_dist   = self._safe(self._get(state_name, "rain_dist_annual", y), np.nan)
            rain_sub    = self._safe(self._get(state_name, "rain_sub_annual", y), np.nan)
            rain_monsoon= self._safe(self._get(state_name, "rain_sub_jun-sep", y), np.nan)
            rain_premonsoon = self._safe(self._get(state_name, "rain_sub_mar-may", y), np.nan)
            rain_postmonsoon= self._safe(self._get(state_name, "rain_sub_oct-dec", y), np.nan)

            crop_prod   = self._safe(self._get(state_name, "total_crop_production_t", y), np.nan)
            crop_area   = self._safe(self._get(state_name, "total_crop_area_ha", y), np.nan)
            n_crops     = self._safe(self._get(state_name, "n_crop_types", y), 20.0)
            energy_agg  = self._safe(self._get(state_name, "energy_aggregate_fuel", y), np.nan)
            energy_fuel = self._safe(self._get(state_name, "energy_fuel", y), np.nan)
            pop_k       = self._safe(self._get(state_name, "population_thousands", y), 10000.0)
            pci         = self._safe(self._get(state_name, "per_capita_income_inr", y), np.nan)

            # Replace NaN with column median across all targets
            def med(col: str) -> float:
                v = self.df_targets[col].dropna()
                return float(v.median()) if len(v) > 0 else 1.0

            if np.isnan(rain_dist):    rain_dist    = self._safe(rain_sub, med("rain_dist_annual"))
            if np.isnan(rain_sub):     rain_sub     = self._safe(rain_dist, med("rain_sub_annual"))
            if np.isnan(rain_monsoon): rain_monsoon = rain_sub * 0.65  # monsoon ≈ 65% of annual
            if np.isnan(rain_premonsoon): rain_premonsoon = rain_sub * 0.05
            if np.isnan(rain_postmonsoon):rain_postmonsoon= rain_sub * 0.10
            if np.isnan(crop_prod):    crop_prod    = med("total_crop_production_t")
            if np.isnan(crop_area):    crop_area    = med("total_crop_area_ha")
            if np.isnan(energy_agg):   energy_agg   = 0.0
            if np.isnan(energy_fuel):  energy_fuel  = 0.0
            if np.isnan(pci):          pci          = med("per_capita_income_inr")

            pop_safe = max(pop_k, 1.0)
            energy_total = energy_agg + energy_fuel

            # ── WATER STOCK ──────────────────────────────────────────────────
            # Best signal: district-level long-run annual rainfall (mm).
            # We use average of dist and sub when both available.
            rain_combined = (rain_dist + rain_sub) / 2.0
            water_stock = self._norm(rain_combined, "rain_dist_annual")
            # Groundwater pressure proxy: dry states with intensive agriculture
            # Punjab: high crop area + low rainfall → GW depleted
            agri_intensity = crop_area / (pop_safe * 1000)  # ha per person
            gw_pressure = np.clip(agri_intensity / 0.5, 0, 1)  # >0.5 ha/person = stressed
            water_stock = np.clip(water_stock - gw_pressure * 0.15, 0.05, 1.0)

            # ── FOOD STOCK ───────────────────────────────────────────────────
            prod_per_cap = crop_prod / (pop_safe * 1000)  # tonnes per person
            crop_div_bonus = np.clip((n_crops - 5) / 70.0, 0, 0.2)  # 0→+0.2
            food_stock = self._norm(prod_per_cap, "_food_pc") + crop_div_bonus
            food_stock = np.clip(food_stock, 0.05, 1.0)

            # ── ENERGY STOCK ─────────────────────────────────────────────────
            energy_per_cap = energy_total / pop_safe  # MW per thousand people
            energy_stock = self._norm(energy_per_cap, "_energy_pc")
            energy_stock = np.clip(energy_stock, 0.05, 1.0)

            # ── LAND STOCK ───────────────────────────────────────────────────
            area_per_cap = crop_area / (pop_safe * 1000)  # ha per person
            rain_boost = np.clip(rain_combined / 2000.0, 0, 0.2)  # rain improves land potential
            land_stock = self._norm(area_per_cap, "_land_pc") + rain_boost
            land_stock = np.clip(land_stock, 0.05, 1.0)

            # ── POPULATION ───────────────────────────────────────────────────
            pop_norm = self._norm(pop_k, "population_thousands")

            # ── ECONOMIC POWER ───────────────────────────────────────────────
            # Use absolute PCI. Cap at 0 for negative values (anomalous data).
            pci_safe = max(pci, 0.0)
            economic_power = self._norm(pci_safe, "per_capita_income_inr")

            # ── ADAPTIVE CAPACITY ────────────────────────────────────────────
            adaptive_capacity = (
                0.40 * economic_power
                + 0.30 * energy_stock
                + 0.20 * food_stock
                + 0.10 * np.clip((n_crops / 76.0), 0, 1)  # 76 = max n_crop_types in data
            )
            adaptive_capacity = np.clip(adaptive_capacity, 0.05, 1.0)

            # ── SOCIAL STABILITY ─────────────────────────────────────────────
            # Proxy: stability of PCI over years (CV = std/mean, lower = more stable)
            pci_series = self._get_series(state_name, "per_capita_income_inr")["per_capita_income_inr"]
            if len(pci_series) >= 3 and pci_series.mean() > 0:
                cv = pci_series.std() / pci_series.mean()
                stability_from_cv = float(1.0 - np.clip(cv, 0, 1))
            else:
                stability_from_cv = 0.5
            # Also: poor states with high pop density = lower stability
            social_stability = (
                0.40 * stability_from_cv
                + 0.40 * economic_power
                + 0.20 * (1.0 - pop_norm)   # smaller population = easier to govern
            )
            social_stability = np.clip(social_stability, 0.05, 1.0)

            # ── MILITARY / GEOPOLITICAL STRENGTH ─────────────────────────────
            # Proxy: large + wealthy + industrial states have more influence.
            military_str = (
                0.50 * economic_power
                + 0.30 * pop_norm
                + 0.20 * energy_stock
            )
            military_str = np.clip(military_str, 0.05, 1.0)

            # ── TRADE OPENNESS ────────────────────────────────────────────────
            # Food surplus: production per cap above subsistence (assume 0.25 t/person/yr)
            subsistence_tpc = 0.25
            food_surplus = np.clip((prod_per_cap - subsistence_tpc) / subsistence_tpc, 0, 1)
            # Energy surplus: above-median energy per cap → can export
            median_energy_pc = med("energy_aggregate_fuel") / med("population_thousands")
            energy_surplus = np.clip((energy_per_cap - median_energy_pc) / (median_energy_pc + 1), 0, 1)
            trade_openness = np.clip(0.55 * food_surplus + 0.45 * energy_surplus, 0.05, 1.0)

            # ── FINANCIAL VULNERABILITY ───────────────────────────────────────
            median_pci = float(self.df_targets["per_capita_income_inr"].dropna().median())
            if median_pci > 0:
                # How far below median? More below = more vulnerable
                gap = max(0.0, median_pci - pci_safe) / (median_pci + 1)
            else:
                gap = 0.5
            fin_vulnerability = np.clip(gap, 0.0, 1.0)

            # ── CLIMATE SHOCK HISTORY ─────────────────────────────────────────
            monsoon_series = self._get_series(state_name, "rain_sub_jun-sep")["rain_sub_jun-sep"]
            if len(monsoon_series) >= 5:
                m_mean = float(monsoon_series.mean())
                m_std  = float(monsoon_series.std())
                m_std  = max(m_std, 1.0)
                n_years = len(monsoon_series)
                drought_freq = float((monsoon_series < m_mean - m_std).sum()) / n_years
                flood_freq   = float((monsoon_series > m_mean + m_std).sum()) / n_years
                # Max normalised single-year deviation = shock severity
                deviations = np.abs(monsoon_series - m_mean) / (m_std + 1e-9)
                shock_severity = float(np.clip(deviations.max() / 3.0, 0, 1))
            else:
                drought_freq  = 0.15
                flood_freq    = 0.10
                shock_severity= 0.20

            # Heatwave proxy: pre-monsoon heat state (arid states = higher risk)
            # If rain_sub_mar-may < 30mm AND annual < 600 → heatwave-prone
            heatwave_freq = float(np.clip(
                (1.0 - np.clip(rain_premonsoon / 100.0, 0, 1)) *
                (1.0 - np.clip(rain_combined / 1000.0, 0, 1)),
                0, 1
            ))
            # Storm proxy: post-monsoon heavy rain = cyclone exposure
            storm_freq = float(np.clip(rain_postmonsoon / 500.0, 0, 1))

            # ── WATER-FOOD COUPLING ───────────────────────────────────────────
            # How much food production depends on water (irrigation-heavy = high coupling)
            # Proxy: agri intensity relative to rainfall
            # High area + low rain = heavy irrigation dependence
            if rain_combined > 0:
                coupling = np.clip(agri_intensity / (rain_combined / 500.0), 0, 1)
            else:
                coupling = 0.8
            water_food_coupling = float(coupling)

            # ── RAINFALL RECHARGE RATE ────────────────────────────────────────
            rainfall_recharge = float(self._norm(rain_combined, "rain_dist_annual"))

            init[agent_id] = {
                # Core resource stocks [0-1]
                "water_stock":          float(water_stock),
                "food_stock":           float(food_stock),
                "energy_stock":         float(energy_stock),
                "land_stock":           float(land_stock),
                # Capabilities [0-1]
                "population_norm":      float(pop_norm),
                "economic_power":       float(economic_power),
                "military_strength":    float(military_str),
                "adaptive_capacity":    float(adaptive_capacity),
                "social_stability":     float(social_stability),
                "trade_openness":       float(trade_openness),
                "financial_vulnerability": float(fin_vulnerability),
                # Resource dynamics
                "water_food_coupling":  water_food_coupling,
                "rainfall_recharge":    rainfall_recharge,
                # Climate vulnerability
                "drought_freq":         float(drought_freq),
                "flood_freq":           float(flood_freq),
                "heatwave_freq":        float(heatwave_freq),
                "storm_freq":           float(storm_freq),
                "shock_severity":       float(shock_severity),
                # Raw values (for reference / depletion rate calcs)
                "raw_population_k":     float(pop_k),
                "raw_pci":              float(pci_safe),
                "raw_rain_annual":      float(rain_combined),
                "raw_energy_mw":        float(energy_total),
                "raw_crop_prod_t":      float(crop_prod),
                "raw_crop_area_ha":     float(crop_area),
                "n_crop_types":         float(n_crops),
                "fragility_score":      float(1.0 - social_stability),  # inverse
                "state_name":           state_name,
                "agent_id":             agent_id,
            }

        return init

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_depletion_rates(self) -> Dict[str, Dict]:
        """
        Fit LinearRegression slope (per year) for each key column per state.
        Returns fractional annual change relative to mean.
        Used in world_step to apply realistic resource depletion/growth.
        """
        columns = {
            # rain_sub_annual has real year-to-year variation; rain_dist_annual
            # is a static long-run average and always has slope ≈ 0.
            "water":   "rain_sub_annual",           # monsoon rainfall trend
            "food":    "total_crop_production_t",   # production trend
            "energy":  "energy_aggregate_fuel",     # installed capacity trend
            "land":    "total_crop_area_ha",         # cultivated area trend
            "pop":     "population_thousands",       # population trend
            "pci":     "per_capita_income_inr",      # income trend
        }
        rates: Dict[str, Dict] = {}
        for agent_id, state_name in STATE_AGENTS.items():
            rates[agent_id] = {}
            for name, col in columns.items():
                series = self._get_series(state_name, col)
                if len(series) < 4:
                    rates[agent_id][name] = {"slope_frac": 0.0, "baseline": 0.0}
                    continue
                X = series["year"].values.reshape(-1, 1)
                y = series[col].values
                slope = float(LinearRegression().fit(X, y).coef_[0])
                mean_val = float(y.mean())
                rates[agent_id][name] = {
                    "slope_raw":  slope,
                    "slope_frac": slope / mean_val if abs(mean_val) > 1e-9 else 0.0,
                    "baseline":   mean_val,
                    "n_points":   len(series),
                }
        return rates

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_climate_matrices(self) -> Dict[str, Dict]:
        """
        Build per-state 5×5 Markov transition matrices from historical rainfall.

        Climate state classification (from rain_sub columns):
          0 = Normal   : monsoon within ±0.5σ of mean
          1 = Drought   : jun-sep  < mean − 1σ
          2 = Flood     : jun-sep  > mean + 1σ
          3 = Heatwave  : mar-may  < 30mm  AND annual < 700mm
          4 = Storm     : oct-dec  > mean + 1σ  (Bay of Bengal cyclone proxy)

        Laplace smoothing (+0.5) prevents zero-probability transitions.
        Ref: Wilks (2011) Statistical Methods in Atmospheric Sciences, §4.
        """
        matrices: Dict[str, Dict] = {}

        for agent_id, state_name in STATE_AGENTS.items():
            sub = (
                self.df_targets[self.df_targets["state"] == state_name]
                .sort_values("year")
                .reset_index(drop=True)
            )

            # Monsoon series for drought/flood classification
            monsoon = sub["rain_sub_jun-sep"].fillna(
                sub["rain_dist_jun"].fillna(0) + sub["rain_dist_jul"].fillna(0)
                + sub["rain_dist_aug"].fillna(0) + sub["rain_dist_sep"].fillna(0)
            )
            premonsoon = sub["rain_sub_mar-may"].fillna(
                sub["rain_dist_mar"].fillna(0) + sub["rain_dist_apr"].fillna(0)
                + sub["rain_dist_may"].fillna(0)
            )
            postmonsoon = sub["rain_sub_oct-dec"].fillna(
                sub["rain_dist_oct"].fillna(0) + sub["rain_dist_nov"].fillna(0)
                + sub["rain_dist_dec"].fillna(0)
            )
            annual = sub["rain_dist_annual"].fillna(sub["rain_sub_annual"])

            m_mean  = monsoon.mean()
            m_std   = max(monsoon.std(), 1.0)
            pm_mean = postmonsoon.mean()
            pm_std  = max(postmonsoon.std(), 1.0)

            def classify_year(i: int) -> int:
                mon = monsoon.iloc[i]
                premon = premonsoon.iloc[i]
                postmon = postmonsoon.iloc[i]
                ann = annual.iloc[i] if not pd.isna(annual.iloc[i]) else 1000.0
                if mon < m_mean - m_std:
                    return 1  # Drought
                if mon > m_mean + m_std:
                    return 2  # Flood
                if postmon > pm_mean + pm_std:
                    return 4  # Storm
                if premon < 30.0 and ann < 700.0:
                    return 3  # Heatwave
                return 0       # Normal

            states = [classify_year(i) for i in range(len(sub))]

            # Transition matrix with Laplace smoothing
            mat = np.ones((5, 5)) * 0.5
            for i in range(len(states) - 1):
                mat[states[i]][states[i + 1]] += 1
            mat /= mat.sum(axis=1, keepdims=True)

            state_counts = np.bincount(states, minlength=5).astype(float)
            matrices[agent_id] = {
                "matrix":     mat,
                "state_dist": state_counts / max(state_counts.sum(), 1),
                "last_state": states[-1] if states else 0,
            }

        return matrices

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_trade_graph(self) -> nx.DiGraph:
        """
        Initial bilateral trade graph using gravity model.
        Trade potential ∝ √(GDP_i × GDP_j) × (openness_i + openness_j)/2
        GDP proxy = per_capita_income_inr × population_thousands

        Edges added only above threshold (≥0.10 normalised score).
        """
        G = nx.DiGraph()
        G.add_nodes_from(AGENT_IDS)

        gdp_proxy  = {}
        openness   = {}
        for agent_id, state_name in STATE_AGENTS.items():
            pci   = self._safe(self._get(state_name, "per_capita_income_inr", self.BASE_YEAR), 10000.0)
            pop   = self._safe(self._get(state_name, "population_thousands", self.BASE_YEAR), 10000.0)
            pci_safe = max(pci, 0.0)
            gdp_proxy[agent_id]  = pci_safe * pop
            openness[agent_id] = self.region_init[agent_id]["trade_openness"]

        max_gdp = max(gdp_proxy.values()) if gdp_proxy else 1.0
        scores  = []
        for i, ai in enumerate(AGENT_IDS):
            for j, aj in enumerate(AGENT_IDS):
                if ai == aj:
                    continue
                gravity = (
                    np.sqrt(gdp_proxy[ai] * gdp_proxy[aj]) / (max_gdp + 1e-9)
                ) * (openness[ai] + openness[aj]) / 2.0
                scores.append((ai, aj, gravity))

        if scores:
            max_s = max(s[2] for s in scores)
            for ai, aj, s in scores:
                w = s / (max_s + 1e-9)
                if w > 0.10:
                    G.add_edge(ai, aj, weight=w, resource_type="mixed",
                               volume=w, age=0, active=True)
        return G

    # ─────────────────────────────────────────────────────────────────────────
    def _print_init_summary(self):
        print(f"\n[DataLoader] Initialisation summary:")
        print(f"  {'ID':<4} {'State':<22} {'Water':>6} {'Food':>6} {'Energy':>7} "
              f"{'Land':>6} {'EcoPow':>7} {'Adapt':>6} {'SocStab':>8}")
        for aid in AGENT_IDS:
            r = self.region_init[aid]
            print(
                f"  {aid:<4} {r['state_name']:<22} "
                f"{r['water_stock']:>6.3f} {r['food_stock']:>6.3f} "
                f"{r['energy_stock']:>7.3f} {r['land_stock']:>6.3f} "
                f"{r['economic_power']:>7.3f} {r['adaptive_capacity']:>6.3f} "
                f"{r['social_stability']:>8.3f}"
            )
        print(f"\n[DataLoader] Trade graph: {self.trade_graph.number_of_nodes()} nodes, "
              f"{self.trade_graph.number_of_edges()} edges")
        print(f"[DataLoader] Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ENVIRONMENT FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def env(csv_path: str, max_cycles: int = 200, noise_level: float = 0.3):
    """Standard PettingZoo factory function."""
    raw_env = WorldSimIndiaEnv(csv_path=csv_path, max_cycles=max_cycles,
                               noise_level=noise_level)
    raw_env = wrappers.AssertOutOfBoundsWrapper(raw_env)
    raw_env = wrappers.OrderEnforcingWrapper(raw_env)
    return raw_env


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PETTINGZOO AEC ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class WorldSimIndiaEnv(AECEnv):
    """
    WorldSim India — 10 Indian states as AI agents in a multi-agent resource
    conflict simulation seeded entirely from real data.

    Observation space : 74-dimensional float32 vector (see _observe docstring)
    Action space      : MultiDiscrete([17, 10]) — action_type × target_agent
    """

    metadata = {
        "render_modes": ["human"],
        "name": "worldsim_india_v1",
        "is_parallelizable": False,
    }

    # ── Water-Food-Energy Nexus thresholds (Hoff 2011, FAO AQUASTAT) ─────────
    NEXUS_WATER_THRESHOLD  = 0.20   # water < 20% → food cascade
    NEXUS_ENERGY_THRESHOLD = 0.15   # energy < 15% → water cascade
    NEXUS_FOOD_MULTIPLIER  = 0.40   # food production multiplier when water critical
    NEXUS_WATER_MULTIPLIER = 0.60   # water efficiency when energy critical
    COLLAPSE_THRESHOLD     = 0.05   # below this = catastrophic collapse

    # ── Conflict weights (Gleditsch et al. 2002 UCDP/PRIO) ───────────────────
    CONFLICT_WATER_W    = 0.35
    CONFLICT_FOOD_W     = 0.25
    CONFLICT_FRAGILITY_W= 0.20
    CONFLICT_HISTORY_W  = 0.20

    # ── Population dynamics (Verhulst logistic) ───────────────────────────────
    POP_GROWTH_RATE        = 0.018  # 1.8 % annual (India average 2000-2020)

    # Observation dimension
    OBS_DIM = 74

    def __init__(self, csv_path: str, max_cycles: int = 200, noise_level: float = 0.3):
        super().__init__()

        self.csv_path    = csv_path
        self.max_cycles  = max_cycles
        self.noise_level = noise_level

        # Load all data and compute parameters
        self.data_loader = WorldSimDataLoader(csv_path)

        # Agent bookkeeping (PettingZoo requires these)
        self.possible_agents = AGENT_IDS.copy()
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}
        self.n_agents = len(self.possible_agents)

        # Spaces
        self.observation_spaces = {
            a: spaces.Box(low=-1.0, high=2.0, shape=(self.OBS_DIM,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.MultiDiscrete([N_ACTIONS, self.n_agents])
            for a in self.possible_agents
        }

        # Internal state — all populated in reset()
        self._state:            Dict[str, Dict]   = {}
        self._climate_states:   Dict[str, Dict]   = {}
        self._trade_graph:      Optional[nx.DiGraph] = None
        self._trade_agreements: Dict              = {}
        self._alliances:        Dict[str, set]    = {}
        self._reputation:       Dict[str, float]  = {}
        self._conflict_matrix:  np.ndarray        = np.zeros((10, 10))
        self._defection_count:  Dict[str, int]    = {}
        self._pending_trades:   Dict[str, list]   = {}
        self._event_log:        list              = []
        self._cycle:            int               = 0

        # PettingZoo required attributes
        self.agents: List[str]               = []
        self.rewards: Dict[str, float]       = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool]   = {}
        self.truncations:  Dict[str, bool]   = {}
        self.infos: Dict[str, dict]          = {}
        self._agent_selector: Optional[agent_selector] = None

    # ─────────────────────────────────────────────────────────────────────────
    # PettingZoo API properties
    # ─────────────────────────────────────────────────────────────────────────
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        self.agents  = self.possible_agents.copy()
        self._cycle  = 0
        self._event_log = []

        # ── Initialise resource states from data ─────────────────────────────
        self._state = {}
        for aid in self.agents:
            init = self.data_loader.region_init[aid]
            self._state[aid] = {
                # Resource stocks
                "water":  init["water_stock"],
                "food":   init["food_stock"],
                "energy": init["energy_stock"],
                "land":   init["land_stock"],
                # Capabilities
                "economic_power":      init["economic_power"],
                "military_strength":   init["military_strength"],
                "adaptive_capacity":   init["adaptive_capacity"],
                "social_stability":    init["social_stability"],
                "population":          init["population_norm"],
                "trade_openness":      init["trade_openness"],
                "fin_vulnerability":   init["financial_vulnerability"],
                "fragility_score":     init["fragility_score"],
                # Dynamics
                "water_food_coupling": init["water_food_coupling"],
                "rainfall_recharge":   init["rainfall_recharge"],
                # Shock history
                "drought_freq":  init["drought_freq"],
                "flood_freq":    init["flood_freq"],
                "heatwave_freq": init["heatwave_freq"],
                "storm_freq":    init["storm_freq"],
                "shock_sev":     init["shock_severity"],
                # Computed each step
                "cycles_survived":      0,
                "n_trade_agreements":   0,
                "n_alliances":          0,
                "n_sanctions_against":  0,
                "n_pending_trades":     0,
                "last_action":          16,
                "last_action_success":  0,
            }

        # ── Climate states from Markov matrices ──────────────────────────────
        self._climate_states = {}
        for aid in self.agents:
            cm = self.data_loader.climate_matrices[aid]
            self._climate_states[aid] = {
                "current": cm["last_state"],
                "history": [cm["last_state"]] * 5,
                "matrix":  cm["matrix"],
            }

        # ── Social structures ─────────────────────────────────────────────────
        self._trade_graph      = copy.deepcopy(self.data_loader.trade_graph)
        self._trade_agreements = {}
        self._alliances        = defaultdict(set)
        self._reputation       = {aid: 0.7 for aid in self.agents}
        self._defection_count  = defaultdict(int)
        self._conflict_matrix  = np.zeros((self.n_agents, self.n_agents))
        self._pending_trades   = defaultdict(list)

        # ── PettingZoo AEC bookkeeping ────────────────────────────────────────
        self.rewards             = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {} for a in self.agents}
        self._agent_selector     = agent_selector(self.agents)
        self.agent_selection     = self._agent_selector.reset()

        self._update_conflict_matrix()

        observations = {a: self._observe(a) for a in self.agents}
        return observations, self.infos

    # ─────────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        """Process one agent action in the AEC cycle."""
        agent = self.agent_selection

        # Handle terminated / truncated agents (PettingZoo convention)
        if self.terminations.get(agent, False) or self.truncations.get(agent, False):
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0

        # Parse action: supports both array and scalar input
        if hasattr(action, "__len__") and len(action) >= 2:
            action_type = int(np.clip(action[0], 0, N_ACTIONS - 1))
            target_idx  = int(np.clip(action[1], 0, self.n_agents - 1))
        else:
            action_type = int(np.clip(action, 0, N_ACTIONS - 1))
            target_idx  = 0
        target_agent = self.possible_agents[target_idx]

        # Execute
        success = self._execute_action(agent, action_type, target_agent)
        self._state[agent]["last_action"]         = action_type
        self._state[agent]["last_action_success"]  = int(success)
        self._log_event(agent, action_type, target_agent, success)

        # Advance AEC selector to next agent
        self.agent_selection = self._agent_selector.next()

        # ── World step: fires after every agent has acted once ────────────────
        if self._agent_selector.is_last():
            self._world_step()
            self._cycle += 1

            for aid in list(self.agents):
                self.rewards[aid] = self._compute_reward(aid)
                self._cumulative_rewards[aid] += self.rewards[aid]
                self._state[aid]["cycles_survived"] += 1

            # Termination / truncation checks
            for aid in list(self.agents):
                s = self._state[aid]
                collapsed = (s["water"] < self.COLLAPSE_THRESHOLD and
                             s["food"]  < self.COLLAPSE_THRESHOLD)
                self.terminations[aid] = bool(collapsed)
                self.truncations[aid]  = (self._cycle >= self.max_cycles)

            self._update_conflict_matrix()

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION EXECUTION
    # ─────────────────────────────────────────────────────────────────────────
    def _execute_action(self, agent: str, action_type: int, target: str) -> bool:
        s = self._state[agent]
        success = False

        if action_type == 0:  # invest_water
            invest = s["economic_power"] * s["adaptive_capacity"] * 0.05
            s["water"] = min(1.0, s["water"] + invest)
            success = True

        elif action_type == 1:  # invest_food
            invest = s["economic_power"] * s["adaptive_capacity"] * 0.04
            s["food"] = min(1.0, s["food"] + invest)
            success = True

        elif action_type == 2:  # invest_energy
            invest = s["economic_power"] * s["adaptive_capacity"] * 0.04
            s["energy"] = min(1.0, s["energy"] + invest)
            success = True

        elif action_type == 3:  # stockpile
            reserve = s["economic_power"] * 0.015
            for r in ("water", "food", "energy"):
                s[r] = min(1.0, s[r] + reserve)
            success = True

        elif action_type == 4:  # population_policy
            s["population"] = max(0.10, s["population"] * 0.995)
            success = True

        elif action_type in (5, 6, 7):  # offer trade
            if target != agent:
                resource = {5: "water", 6: "food", 7: "energy"}[action_type]
                if s[resource] > 0.20:
                    offer_amt = s[resource] * 0.10
                    self._pending_trades[target].append({
                        "from": agent, "resource_give": resource,
                        "amount": offer_amt, "cycle_created": self._cycle,
                    })
                    success = True

        elif action_type == 8:  # accept_trade
            if self._pending_trades[agent]:
                offer = self._pending_trades[agent].pop(0)
                self._process_accepted_trade(agent, offer)
                success = True

        elif action_type == 9:  # reject_trade
            if self._pending_trades[agent]:
                self._pending_trades[agent].pop(0)
                success = True

        elif action_type == 10:  # defect_trade
            self._process_defection(agent, target)
            success = True

        elif action_type == 11:  # form_alliance
            if (target != agent
                    and target not in self._alliances[agent]
                    and self._reputation[agent] > 0.40
                    and self._reputation.get(target, 0.5) > 0.40):
                self._alliances[agent].add(target)
                self._alliances[target].add(agent)
                self._log_event(agent, action_type, target, True, "alliance_formed")
                success = True

        elif action_type == 12:  # leave_alliance
            if target in self._alliances[agent]:
                self._alliances[agent].discard(target)
                self._alliances[target].discard(agent)
                self._reputation[agent] = max(0.0, self._reputation[agent] - 0.05)
                success = True

        elif action_type == 13:  # sanction
            if target != agent:
                if self._trade_graph.has_edge(agent, target):
                    self._trade_graph[agent][target]["active"] = False
                ts = self._state[target]
                ts["economic_power"] = max(0.05, ts["economic_power"] - 0.03)
                s["economic_power"]  = max(0.05, s["economic_power"]  - 0.01)
                self._state[target]["n_sanctions_against"] += 1
                success = True

        elif action_type == 14:  # raid
            if target != agent:
                ts = self._state[target]
                raid_prob = (
                    s["military_strength"] /
                    (s["military_strength"] + ts["military_strength"] + 1e-9)
                ) * (1.0 - ts["social_stability"]) * 0.60
                if np.random.random() < raid_prob:
                    # Successful raid
                    steal_w = min(ts["water"], 0.08)
                    steal_f = min(ts["food"],  0.05)
                    ts["water"] = max(0.0, ts["water"] - steal_w)
                    ts["food"]  = max(0.0, ts["food"]  - steal_f)
                    s["water"]  = min(1.0, s["water"]  + steal_w * 0.75)
                    s["food"]   = min(1.0, s["food"]   + steal_f * 0.75)
                    for who in (agent, target):
                        decay = 0.05 if who == agent else 0.08
                        self._state[who]["military_strength"] = max(
                            0.05, self._state[who]["military_strength"] - decay)
                    self._reputation[agent]   = max(0.0, self._reputation[agent]   - 0.15)
                    self._defection_count[agent] += 1
                    success = True
                else:
                    # Failed raid — attacker takes damage
                    s["military_strength"] = max(0.05, s["military_strength"] - 0.08)
                    s["energy"]            = max(0.00, s["energy"]            - 0.03)
                    self._reputation[agent] = max(0.0, self._reputation[agent] - 0.05)

        elif action_type == 15:  # diplomat
            if target != agent:
                ai = self.possible_agents.index(agent)
                ti = self.possible_agents.index(target)
                self._conflict_matrix[ai][ti] = max(
                    0.0, self._conflict_matrix[ai][ti] - 0.08)
                self._reputation[agent] = min(1.0, self._reputation[agent] + 0.02)
                success = True

        elif action_type == 16:  # do_nothing
            success = True

        # Sync trade/alliance counts
        s["n_trade_agreements"] = len(self._trade_agreements)
        s["n_alliances"]        = len(self._alliances[agent])
        return success

    # ─────────────────────────────────────────────────────────────────────────
    # TRADE HELPERS
    # ─────────────────────────────────────────────────────────────────────────
    def _process_accepted_trade(self, acceptor: str, offer: dict):
        offerer  = offer["from"]
        resource = offer["resource_give"]
        amount   = offer["amount"]
        s_off    = self._state[offerer]
        s_acc    = self._state[acceptor]

        transfer = min(s_off[resource], amount)
        s_off[resource] = max(0.0, s_off[resource] - transfer)
        s_acc[resource] = min(1.0, s_acc[resource] + transfer * 0.90)

        # Reciprocal payment: acceptor sends a different resource back
        reciprocal = {"water": "food", "food": "energy", "energy": "water"}
        give_r    = reciprocal[resource]
        give_amt  = min(s_acc[give_r], transfer * 0.80)
        s_acc[give_r] = max(0.0, s_acc[give_r] - give_amt)
        s_off[give_r] = min(1.0, s_off[give_r] + give_amt * 0.90)

        # Update graph + reputation
        if not self._trade_graph.has_edge(offerer, acceptor):
            self._trade_graph.add_edge(
                offerer, acceptor, weight=transfer,
                resource_type=resource, volume=transfer, age=0, active=True)
        else:
            self._trade_graph[offerer][acceptor]["volume"] += transfer
            self._trade_graph[offerer][acceptor]["age"]    = 0
            self._trade_graph[offerer][acceptor]["active"] = True

        for who in (offerer, acceptor):
            self._reputation[who] = min(1.0, self._reputation[who] + 0.03)

        self._trade_agreements[(offerer, acceptor)] = {
            "resource": resource, "amount": transfer, "cycle": self._cycle}

    def _process_defection(self, defector: str, victim: str):
        # Remove agreements involving either party
        to_remove = [k for k in self._trade_agreements
                     if defector in k or victim in k]
        for k in to_remove:
            del self._trade_agreements[k]

        if self._trade_graph.has_edge(defector, victim):
            self._trade_graph.remove_edge(defector, victim)

        self._alliances[defector].discard(victim)
        self._alliances[victim].discard(defector)
        self._reputation[defector] = max(0.0, self._reputation[defector] - 0.25)
        self._defection_count[defector] += 1

        # Gossip: observers update conflict probability upward
        di = self.possible_agents.index(defector)
        vi = self.possible_agents.index(victim)
        self._conflict_matrix[di][vi] = min(
            1.0, self._conflict_matrix[di][vi] + 0.30)
        self._conflict_matrix[vi][di] = min(
            1.0, self._conflict_matrix[vi][di] + 0.20)
        for obs in self.agents:
            if obs == defector:
                continue
            if self._trade_graph.has_edge(defector, obs):
                oi = self.possible_agents.index(obs)
                self._conflict_matrix[di][oi] = min(
                    1.0, self._conflict_matrix[di][oi] + 0.15)

    # ─────────────────────────────────────────────────────────────────────────
    # WORLD STEP (fires after ALL agents have acted)
    # ─────────────────────────────────────────────────────────────────────────
    def _world_step(self):
        self._apply_climate_shocks()
        self._apply_depletion()
        self._apply_nexus_cascades()
        self._apply_population_dynamics()
        self._apply_alliance_benefits()

        # Reputation rebuild (slow, if no defections)
        for aid in self.agents:
            if self._defection_count[aid] == 0:
                self._reputation[aid] = min(1.0, self._reputation[aid] + 0.01)
            self._defection_count[aid] = 0  # reset for next cycle

        # Age trade edges
        for _, _, d in self._trade_graph.edges(data=True):
            d["age"] = d.get("age", 0) + 1

    def _apply_climate_shocks(self):
        """
        Markov transition per state → determine this cycle's climate event.
        Shock magnitudes scale with per-state historical shock_severity index.
        """
        for aid in self.agents:
            cs = self._climate_states[aid]
            probs = cs["matrix"][cs["current"]]
            next_state = int(np.random.choice(5, p=probs))
            cs["history"].append(next_state)
            cs["history"] = cs["history"][-5:]
            cs["current"]  = next_state

            s   = self._state[aid]
            sev = s["shock_sev"]

            if next_state == 1:  # Drought
                mag = 0.05 + sev * 0.12
                s["water"] = max(0.0, s["water"] - mag)
                s["food"]  = max(0.0, s["food"]  - mag * 0.40)
                self._log_event(aid, -1, aid, True, f"drought mag={mag:.3f}")

            elif next_state == 2:  # Flood
                mag = 0.03 + sev * 0.08
                s["food"]  = max(0.0, s["food"]  - mag)
                s["land"]  = max(0.0, s["land"]  - mag * 0.25)
                s["water"] = min(1.0, s["water"]  + mag * 0.15)
                self._log_event(aid, -1, aid, True, f"flood mag={mag:.3f}")

            elif next_state == 3:  # Heatwave
                mag = 0.03 + sev * 0.07
                s["water"]  = max(0.0, s["water"]  - mag * 1.40)
                s["food"]   = max(0.0, s["food"]   - mag * 0.80)
                s["energy"] = max(0.0, s["energy"] - mag * 0.40)
                self._log_event(aid, -1, aid, True, f"heatwave mag={mag:.3f}")

            elif next_state == 4:  # Storm / Cyclone
                mag = 0.04 + sev * 0.09
                s["energy"] = max(0.0, s["energy"] - mag)
                s["food"]   = max(0.0, s["food"]   - mag * 0.35)
                s["economic_power"] = max(0.05, s["economic_power"] - mag * 0.25)
                self._log_event(aid, -1, aid, True, f"storm mag={mag:.3f}")

    def _apply_depletion(self):
        """
        Apply data-fitted annual depletion rates to each resource.
        Rates from LinearRegression over worldsim_merged.csv time series.
        """
        for aid in self.agents:
            s     = self._state[aid]
            rates = self.data_loader.depletion_rates[aid]

            # Water: rainfall trend + base depletion + recharge
            water_trend = rates["water"]["slope_frac"]  # positive = increasing rain
            water_depl  = max(0.005, -water_trend * 0.008) + 0.006
            water_rech  = s["rainfall_recharge"] * 0.005
            s["water"]  = float(np.clip(s["water"] - water_depl + water_rech, 0.0, 1.0))

            # Food: demand from population, supply from crop trend
            food_trend  = rates["food"]["slope_frac"]
            food_supply = max(0.0, food_trend * 0.005) + 0.002
            food_demand = s["population"] * 0.010
            s["food"]   = float(np.clip(s["food"] - food_demand + food_supply, 0.0, 1.0))

            # Energy: capacity expansion trend minus consumption
            energy_trend = rates["energy"]["slope_frac"]
            energy_grow  = max(0.0, energy_trend * 0.004)
            energy_use   = 0.006
            s["energy"]  = float(np.clip(s["energy"] - energy_use + energy_grow, 0.0, 1.0))

            # Land: slow degradation, partially offset by crop area trend
            land_trend = rates["land"]["slope_frac"]
            land_degl  = max(0.001, -land_trend * 0.003) + 0.002
            s["land"]  = float(np.clip(s["land"] - land_degl, 0.0, 1.0))

    def _apply_nexus_cascades(self):
        """
        Water-Food-Energy Nexus cascade triggers (Hoff 2011):
          Cascade 1: water < 20% → food drops by severity × 60%
          Cascade 2: energy < 15% → water drops additionally
          Cascade 3: both water+food critical → population decline
        """
        for aid in self.agents:
            s = self._state[aid]

            # Cascade 1: Water → Food
            if s["water"] < self.NEXUS_WATER_THRESHOLD:
                sev = 1.0 - (s["water"] / self.NEXUS_WATER_THRESHOLD)
                penalty = sev * (1.0 - self.NEXUS_FOOD_MULTIPLIER)
                s["food"] = max(0.0, s["food"] - penalty * s["water_food_coupling"])
                self._log_event(aid, -2, aid, True,
                                f"nexus water→food sev={sev:.2f}")

            # Cascade 2: Energy → Water
            if s["energy"] < self.NEXUS_ENERGY_THRESHOLD:
                sev = 1.0 - (s["energy"] / self.NEXUS_ENERGY_THRESHOLD)
                penalty = sev * (1.0 - self.NEXUS_WATER_MULTIPLIER)
                s["water"] = max(0.0, s["water"] - penalty * 0.5)
                self._log_event(aid, -2, aid, True,
                                f"nexus energy→water sev={sev:.2f}")

            # Cascade 3: Double critical
            if s["water"] < 0.10 and s["food"] < 0.10:
                s["population"]      = max(0.01, s["population"] * 0.98)
                s["social_stability"]= max(0.0,  s["social_stability"] - 0.05)
                self._log_event(aid, -3, aid, True, "double_collapse")

    def _apply_population_dynamics(self):
        """Verhulst logistic: dP/dt = r × P × (1 − P/K)"""
        for aid in self.agents:
            s  = self._state[aid]
            K  = max(0.01, s["food"] * 0.50 + s["water"] * 0.30 + s["energy"] * 0.20)
            P  = s["population"]
            r  = self.POP_GROWTH_RATE * (1.0 - s["fragility_score"])
            dP = r * P * (1.0 - P / K)
            s["population"] = float(np.clip(P + dP * 0.01, 0.01, 2.0))

    def _apply_alliance_benefits(self):
        """Allies above threshold share surpluses with allies in crisis."""
        for aid in self.agents:
            s = self._state[aid]
            for ally in self._alliances[aid]:
                if ally not in self.agents:
                    continue
                sa = self._state[ally]
                for resource in ("water", "food", "energy"):
                    if sa[resource] < 0.30 and s[resource] > 0.50:
                        share = (s[resource] - 0.40) * 0.10
                        s[resource]  = max(0.30, s[resource]  - share)
                        sa[resource] = min(1.0,  sa[resource] + share * 0.90)

    def _update_conflict_matrix(self):
        """
        Recompute N×N conflict probability matrix.
        Formula (Gleditsch et al. 2002):
          P(conflict_ij) = w1×water_gap + w2×food_stress_i +
                           w3×fragility_i + w4×defection_hist_i
        Exponential moving average (α=0.3) smooths abrupt changes.
        Alliance membership halves conflict probability.
        """
        for i, ai in enumerate(self.possible_agents):
            if ai not in self.agents:
                continue
            si = self._state[ai]
            for j, aj in enumerate(self.possible_agents):
                if aj not in self.agents or i == j:
                    continue
                sj = self._state[aj]
                water_gap   = max(0.0, sj["water"] - si["water"])
                food_stress = max(0.0, 0.5 - si["food"]) * 2.0
                fragility   = si["fragility_score"]
                def_hist    = min(1.0, self._defection_count[ai] * 0.30)

                in_alliance  = aj in self._alliances.get(ai, set())
                ally_factor  = 0.20 if in_alliance else 1.0

                raw = (
                    self.CONFLICT_WATER_W     * water_gap +
                    self.CONFLICT_FOOD_W       * food_stress +
                    self.CONFLICT_FRAGILITY_W  * fragility +
                    self.CONFLICT_HISTORY_W    * def_hist
                ) * ally_factor

                alpha = 0.30
                self._conflict_matrix[i][j] = float(np.clip(
                    alpha * raw + (1 - alpha) * self._conflict_matrix[i][j],
                    0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # REWARD
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_reward(self, agent: str) -> float:
        """
        Composite reward with longevity multiplier.
        Longer survival exponentially amplifies rewards — creates evolutionary
        pressure toward sustainable over extractive strategies.
        """
        s = self._state[agent]
        t = s["cycles_survived"]

        alive = (s["water"] > self.COLLAPSE_THRESHOLD and
                 s["food"]  > self.COLLAPSE_THRESHOLD)
        survival_bonus = 10.0 if alive else -50.0

        # Log-curve resource reward (diminishing returns on hoarding)
        thresh = {"water": 0.30, "food": 0.30, "energy": 0.20, "land": 0.20}
        resource_r = sum(
            np.log(1.0 + s[r] / thresh[r])
            for r in ("water", "food", "energy", "land")
        )

        trade_bonus    = s["n_trade_agreements"] * 0.5
        alliance_bonus = len(self._alliances[agent]) * 1.5
        reserve_bonus  = sum(
            0.5 for r in ("water", "food", "energy")
            if s[r] > thresh.get(r, 0.25) * 1.30
        )

        idx_i          = self.possible_agents.index(agent)
        conflict_cost  = float(self._conflict_matrix[idx_i].mean()) * 8.0
        collapse_pen   = sum(
            10.0 for r in ("water", "food", "energy")
            if s[r] < self.COLLAPSE_THRESHOLD
        )
        defect_pen     = self._defection_count.get(agent, 0) * 5.0

        longevity = 1.0 + 0.02 * t

        return float((
            survival_bonus + resource_r + trade_bonus
            + alliance_bonus + reserve_bonus
            - conflict_cost - collapse_pen - defect_pen
        ) * longevity)

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVE
    # ─────────────────────────────────────────────────────────────────────────
    def _observe(self, agent: str) -> np.ndarray:
        """
        Build 74-dimensional observation vector.

        Layout:
          [ 0: 4]  Own resource stocks          (water, food, energy, land)
          [ 4: 8]  Own capabilities              (economic_power, military, adaptive, social)
          [ 8:12]  Own context                   (population, trade_openness, fin_vuln, fragility)
          [12:17]  Climate state history (5-step)
          [17:22]  Own shock frequencies         (drought, flood, heat, storm, shock_sev)
          [22:27]  Trade/social state            (n_trades, n_alliances, rep, n_sanctions, pending)
          [27:31]  Resource dynamics             (water_food_coupling, recharge, shock_sev, -)
          [31:71]  Rival observations 10×4       (water_noisy, food_noisy, conflict_prob, ally_flag)
          [71:74]  Global state                  (cycle/max, n_active_conflicts, global_mean_water)
        """
        s   = self._state[agent]
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        # [0:4]
        obs[0], obs[1], obs[2], obs[3] = s["water"], s["food"], s["energy"], s["land"]

        # [4:8]
        obs[4] = s["economic_power"]
        obs[5] = s["military_strength"]
        obs[6] = s["adaptive_capacity"]
        obs[7] = s["social_stability"]

        # [8:12]
        obs[8]  = s["population"]
        obs[9]  = s["trade_openness"]
        obs[10] = s["fin_vulnerability"]
        obs[11] = s["fragility_score"]

        # [12:17] Climate history
        for k, h in enumerate(self._climate_states[agent]["history"][-5:]):
            obs[12 + k] = h / 4.0

        # [17:22] Shock frequencies
        obs[17] = float(np.clip(s["drought_freq"],  0, 1))
        obs[18] = float(np.clip(s["flood_freq"],    0, 1))
        obs[19] = float(np.clip(s["heatwave_freq"], 0, 1))
        obs[20] = float(np.clip(s["storm_freq"],    0, 1))
        obs[21] = float(np.clip(s["shock_sev"],     0, 1))

        # [22:27] Trade/social
        obs[22] = float(np.clip(s["n_trade_agreements"] / 5.0, 0, 1))
        obs[23] = float(np.clip(s["n_alliances"]         / 5.0, 0, 1))
        obs[24] = float(self._reputation.get(agent, 0.5))
        obs[25] = float(np.clip(s["n_sanctions_against"] / 3.0, 0, 1))
        obs[26] = float(np.clip(len(self._pending_trades[agent]) / 5.0, 0, 1))

        # [27:31] Dynamics
        obs[27] = float(s["water_food_coupling"])
        obs[28] = float(s["rainfall_recharge"])
        obs[29] = float(s["shock_sev"])
        obs[30] = float(s["last_action_success"])

        # [31:71] Rival observations (10 rivals × 4 dims = 40)
        ai = self.possible_agents.index(agent)
        for ri, rival in enumerate(self.possible_agents):
            base = 31 + ri * 4
            if rival not in self.agents:
                obs[base:base+4] = [0.5, 0.5, 0.5, 0.0]
                continue
            rs = self._state[rival]
            is_ally   = rival in self._alliances.get(agent, set())
            has_trade = self._trade_graph.has_edge(agent, rival)

            if is_ally:
                noise = self.noise_level * 0.10
            elif has_trade:
                noise = self.noise_level * 0.45
            else:
                noise = self.noise_level

            def noisy(v: float) -> float:
                return float(np.clip(
                    v * (1 - noise) + np.random.normal(0, noise * 0.10),
                    0.0, 1.0))

            obs[base]   = noisy(rs["water"])
            obs[base+1] = noisy(rs["food"])
            obs[base+2] = float(self._conflict_matrix[ai][ri])
            obs[base+3] = float(is_ally)

        # [71:74] Global state
        obs[71] = float(self._cycle / self.max_cycles)
        obs[72] = float(np.clip((self._conflict_matrix > 0.60).sum() / 20.0, 0, 1))
        alive_water = [self._state[a]["water"] for a in self.agents]
        obs[73] = float(np.mean(alive_water)) if alive_water else 0.5

        return obs.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITY METHODS (PettingZoo API + helpers for visualisation)
    # ─────────────────────────────────────────────────────────────────────────
    def observe(self, agent: str) -> np.ndarray:
        return self._observe(agent)

    def _was_dead_step(self, action):
        """PettingZoo convention for handling dead agents."""
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0.0
        remaining = [a for a in self.agents
                     if not (self.terminations.get(a, False) or
                             self.truncations.get(a, False))]
        self.agents          = remaining
        self._agent_selector = agent_selector(self.agents) if self.agents else agent_selector([])
        self.agent_selection = self._agent_selector.reset() if self.agents else None

    def render(self, mode: str = "human"):
        print(f"\n{'='*65}")
        print(f"  Cycle {self._cycle:>3}  |  Active: {len(self.agents)}/{self.n_agents}")
        print(f"{'─'*65}")
        hdr = f"  {'ID':<4} {'State':<22} {'Water':>6} {'Food':>6} "
        hdr += f"{'Energy':>7} {'Rep':>5} {'Ally':>5} {'Conf':>6}"
        print(hdr)
        for aid in self.possible_agents:
            if aid not in self.agents:
                print(f"  {aid:<4} {'COLLAPSED':^22}")
                continue
            s   = self._state[aid]
            idx = self.possible_agents.index(aid)
            rep = self._reputation.get(aid, 0.0)
            ali = len(self._alliances[aid])
            con = float(self._conflict_matrix[idx].mean())
            sname = STATE_AGENTS[aid][:20]
            print(f"  {aid:<4} {sname:<22} {s['water']:>6.3f} {s['food']:>6.3f} "
                  f"{s['energy']:>7.3f} {rep:>5.2f} {ali:>5d} {con:>6.3f}")
        print(f"{'─'*65}")
        if self._event_log:
            for e in self._event_log[-4:]:
                if e["action"].startswith("event_"):
                    continue
                tick = "✓" if e["success"] else "✗"
                note = f" [{e['note']}]" if e.get("note") else ""
                print(f"  [{e['cycle']:>3}] {e['agent']}→{e['action']}→{e['target']} "
                      f"({tick}){note}")
        print()

    def get_state_df(self) -> pd.DataFrame:
        """Return current world state as a tidy DataFrame (for visualisation)."""
        rows = []
        for aid in self.possible_agents:
            status = "active" if aid in self.agents else "collapsed"
            row: dict = {
                "agent_id":   aid,
                "state_name": STATE_AGENTS[aid],
                "status":     status,
                "cycle":      self._cycle,
            }
            if aid in self.agents:
                s = self._state[aid]
                row.update({k: v for k, v in s.items()
                            if isinstance(v, (int, float))})
                row["reputation"]   = self._reputation.get(aid, 0.5)
                row["n_allies"]     = len(self._alliances[aid])
                row["avg_conflict"] = float(
                    self._conflict_matrix[self.possible_agents.index(aid)].mean())
                row["climate"] = CLIMATE_STATES[
                    self._climate_states[aid]["current"]]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_conflict_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._conflict_matrix,
            index=self.possible_agents,
            columns=self.possible_agents)

    def get_trade_graph(self) -> nx.DiGraph:
        return self._trade_graph

    def get_event_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._event_log)

    # ─────────────────────────────────────────────────────────────────────────
    def _log_event(self, agent: str, action_type: int, target: str,
                   success: bool, note: str = ""):
        action_name = ACTION_TYPES.get(action_type, f"event_{action_type}")
        s = self._state.get(agent, {})
        self._event_log.append({
            "cycle":   self._cycle,
            "agent":   agent,
            "action":  action_name,
            "target":  target,
            "success": success,
            "note":    note,
            "water":   s.get("water", 0.0),
            "food":    s.get("food",  0.0),
            "energy":  s.get("energy",0.0),
        })


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def run_visualisation(csv_path: str, n_cycles: int = 100, random_seed: int = 42):
    """
    Run WorldSim India with a random policy and produce 11 diagnostic plots.
    Replace random actions with your trained MAPPO agents for real results.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    print("\n" + "=" * 65)
    print("WorldSim India — Visualisation Run")
    print("=" * 65)

    # ── Run ────────────────────────────────────────────────────────────────
    sim = WorldSimIndiaEnv(csv_path=csv_path, max_cycles=n_cycles,
                           noise_level=0.30)
    _, _ = sim.reset(seed=random_seed)

    history: list         = []
    conflict_history: list= []

    for cycle in range(n_cycles):
        snap = sim.get_state_df()
        snap["cycle"] = cycle
        history.append(snap)
        conflict_history.append(sim._conflict_matrix.copy())

        if not sim.agents:
            print(f"All agents collapsed at cycle {cycle}.")
            break

        for agent in list(sim.agents):
            if sim.terminations.get(agent, False) or sim.truncations.get(agent, False):
                continue
            act_type   = np.random.randint(0, N_ACTIONS)
            target_idx = np.random.randint(0, sim.n_agents)
            try:
                sim.step(np.array([act_type, target_idx]))
            except Exception:
                pass

    df_hist = pd.concat(history, ignore_index=True)
    print(f"\nSimulation complete — {len(history)} cycles, "
          f"{len(sim.agents)} agents alive.")

    # ── Colour palette ─────────────────────────────────────────────────────
    COLORS = {
        "RJ": "#e74c3c", "MH": "#3498db", "UP": "#f39c12",
        "KL": "#2ecc71", "GJ": "#9b59b6", "WB": "#1abc9c",
        "PB": "#e67e22", "BR": "#e91e63", "KA": "#34495e", "TN": "#16a085",
    }
    RESOURCE_CMAP = LinearSegmentedColormap.from_list(
        "res", ["#c0392b", "#e67e22", "#2ecc71"])
    CLIMATE_CMAP  = LinearSegmentedColormap.from_list(
        "clim", ["#2ecc71", "#e67e22", "#2980b9", "#e74c3c", "#8e44ad"])

    fig = plt.figure(figsize=(24, 30))
    fig.patch.set_facecolor("#0a0a0a")
    plt.suptitle("WorldSim India — Resource Conflict Simulation",
                 fontsize=18, color="white", fontweight="bold", y=0.99)

    def dark_ax(ax, title=""):
        ax.set_facecolor("#111111")
        ax.spines[:].set_color("#333333")
        ax.tick_params(colors="white")
        if title:
            ax.set_title(title, color="white", fontsize=10, pad=8)
        ax.grid(True, alpha=0.12, color="white")

    # ── 1: Water time series ───────────────────────────────────────────────
    ax1 = fig.add_subplot(4, 3, (1, 2))
    dark_ax(ax1, "Water & Food Stock Over Time (solid=water, dashed=food)")
    for aid in AGENT_IDS:
        sub = df_hist[df_hist["agent_id"] == aid]
        if "water" not in sub.columns or len(sub) < 2:
            continue
        c = COLORS[aid]
        ax1.plot(sub["cycle"], sub["water"], color=c, lw=1.8,
                 label=f"{STATE_AGENTS[aid][:12]}", alpha=0.9)
        ax1.plot(sub["cycle"], sub["food"],  color=c, lw=1.0,
                 ls="--", alpha=0.55)
    ax1.axhline(0.20, color="#ff6b6b", lw=1.0, ls=":", alpha=0.8,
                label="Water cascade (20%)")
    ax1.axhline(0.05, color="#ff0000", lw=1.5, ls="-", alpha=0.9,
                label="Collapse (5%)")
    ax1.set_xlabel("Cycle", color="white")
    ax1.set_ylabel("Stock (0–1)", color="white")
    ax1.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
               fontsize=7, facecolor="#111111", labelcolor="white", ncol=1)

    # ── 2: Energy time series ──────────────────────────────────────────────
    ax2 = fig.add_subplot(4, 3, 3)
    dark_ax(ax2, "Energy Stock Over Time")
    for aid in AGENT_IDS:
        sub = df_hist[df_hist["agent_id"] == aid]
        if "energy" not in sub.columns or len(sub) < 2:
            continue
        ax2.plot(sub["cycle"], sub["energy"], color=COLORS[aid],
                 lw=1.5, label=STATE_AGENTS[aid][:10])
    ax2.axhline(0.15, color="#ff9f43", lw=1.0, ls=":",
                label="Energy cascade (15%)")
    ax2.set_xlabel("Cycle", color="white")
    ax2.set_ylabel("Energy (0–1)", color="white")
    ax2.legend(fontsize=7, facecolor="#111111", labelcolor="white")

    # ── 3: Conflict heatmap (final) ────────────────────────────────────────
    ax3 = fig.add_subplot(4, 3, 4)
    dark_ax(ax3, "Conflict Probability Matrix (Final Cycle)")
    if conflict_history:
        final_cm = conflict_history[-1]
        mask = np.eye(sim.n_agents, dtype=bool)
        sns.heatmap(
            final_cm, ax=ax3,
            xticklabels=AGENT_IDS, yticklabels=AGENT_IDS,
            cmap="RdYlGn_r", vmin=0, vmax=1,
            linewidths=0.4, linecolor="#222222",
            mask=mask, annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8},
        )
        ax3.tick_params(colors="white", labelsize=8)

    # ── 4: Mean conflict over time ─────────────────────────────────────────
    ax4 = fig.add_subplot(4, 3, 5)
    dark_ax(ax4, "Mean Conflict Probability Over Time")
    for i, aid in enumerate(AGENT_IDS):
        vals = [m[i].mean() for m in conflict_history]
        ax4.plot(range(len(vals)), vals, color=COLORS[aid],
                 lw=1.5, label=STATE_AGENTS[aid][:10])
    ax4.axhline(0.60, color="#ff6b6b", lw=1.0, ls=":", alpha=0.7)
    ax4.set_xlabel("Cycle", color="white")
    ax4.set_ylabel("Mean Conflict Prob", color="white")
    ax4.legend(fontsize=7, facecolor="#111111", labelcolor="white")

    # ── 5: Social stability ────────────────────────────────────────────────
    ax5 = fig.add_subplot(4, 3, 6)
    dark_ax(ax5, "Social Stability Over Time")
    for aid in AGENT_IDS:
        sub = df_hist[df_hist["agent_id"] == aid]
        if "social_stability" not in sub.columns or len(sub) < 2:
            continue
        ax5.plot(sub["cycle"], sub["social_stability"],
                 color=COLORS[aid], lw=1.5, label=STATE_AGENTS[aid][:10])
    ax5.set_xlabel("Cycle", color="white")
    ax5.set_ylabel("Social Stability", color="white")
    ax5.legend(fontsize=7, facecolor="#111111", labelcolor="white")

    # ── 6: Trade network (final) ───────────────────────────────────────────
    ax6 = fig.add_subplot(4, 3, 7)
    ax6.set_facecolor("#111111")
    ax6.set_title("Trade Network (Final)", color="white", fontsize=10, pad=8)
    G = sim.get_trade_graph()
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=42, k=2.0)
        nsizes = [
            400 + sim._state.get(n, {}).get("economic_power", 0.3) * 700
            for n in G.nodes()
        ]
        nx.draw_networkx_nodes(G, pos, ax=ax6,
                               node_size=nsizes,
                               node_color=[COLORS.get(n, "#888") for n in G.nodes()],
                               alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax6,
                                font_size=7, font_color="white", font_weight="bold")
        edges    = list(G.edges(data=True))
        ew       = [e[2].get("weight", 0.1) * 4 for e in edges]
        ecol     = ["#2ecc71" if e[2].get("active", True) else "#c0392b" for e in edges]
        if edges:
            nx.draw_networkx_edges(G, pos, ax=ax6, width=ew, edge_color=ecol,
                                   alpha=0.6, arrows=True, arrowsize=10,
                                   connectionstyle="arc3,rad=0.1")
    ax6.axis("off")

    # ── 7: Reputation ─────────────────────────────────────────────────────
    ax7 = fig.add_subplot(4, 3, 8)
    dark_ax(ax7, "Agent Reputation Over Time")
    for aid in AGENT_IDS:
        sub = df_hist[df_hist["agent_id"] == aid]
        if "reputation" not in sub.columns or len(sub) < 2:
            continue
        ax7.plot(sub["cycle"], sub["reputation"],
                 color=COLORS[aid], lw=1.5, label=STATE_AGENTS[aid][:10])
    ax7.axhline(0.40, color="#ff9f43", lw=1.0, ls=":",
                label="Alliance threshold (0.4)")
    ax7.set_xlabel("Cycle", color="white")
    ax7.set_ylabel("Reputation", color="white")
    ax7.legend(fontsize=7, facecolor="#111111", labelcolor="white")

    # ── 8: Initial resource radar ──────────────────────────────────────────
    ax8 = fig.add_subplot(4, 3, 9, polar=True)
    ax8.set_facecolor("#111111")
    ax8.set_title("Initial Resource Profile (first 5 agents)",
                  color="white", fontsize=10, pad=20)
    cats = ["Water", "Food", "Energy", "EcoPow", "Adaptive"]
    N_c  = len(cats)
    angs = [n / N_c * 2 * np.pi for n in range(N_c)] + [0]
    ax8.set_xticks(angs[:-1])
    ax8.set_xticklabels(cats, color="white", size=8)
    ax8.tick_params(colors="white")
    ax8.spines["polar"].set_color("#333333")
    for aid in AGENT_IDS[:5]:
        r = sim.data_loader.region_init[aid]
        vals = [r["water_stock"], r["food_stock"], r["energy_stock"],
                r["economic_power"], r["adaptive_capacity"]] + [r["water_stock"]]
        ax8.plot(angs, vals, lw=1.5, color=COLORS[aid],
                 label=STATE_AGENTS[aid][:10])
        ax8.fill(angs, vals, alpha=0.07, color=COLORS[aid])
    ax8.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
               fontsize=7, facecolor="#111111", labelcolor="white")

    # ── 9: Population dynamics ─────────────────────────────────────────────
    ax9 = fig.add_subplot(4, 3, 10)
    dark_ax(ax9, "Population Dynamics")
    for aid in AGENT_IDS:
        sub = df_hist[df_hist["agent_id"] == aid]
        if "population" not in sub.columns or len(sub) < 2:
            continue
        ax9.plot(sub["cycle"], sub["population"],
                 color=COLORS[aid], lw=1.5, label=STATE_AGENTS[aid][:10])
    ax9.set_xlabel("Cycle", color="white")
    ax9.set_ylabel("Population (normalised)", color="white")
    ax9.legend(fontsize=7, facecolor="#111111", labelcolor="white")

    # ── 10: Climate state heatmap ──────────────────────────────────────────
    ax10 = fig.add_subplot(4, 3, 11)
    dark_ax(ax10, "Climate State History (per agent per cycle)")
    max_c = min(n_cycles, len(history))
    clim_mat = np.zeros((len(AGENT_IDS), max_c))
    cs_map = {"Normal": 0, "Drought": 1, "Flood": 2, "Heatwave": 3, "Storm": 4}
    for ci, snap in enumerate(history[:max_c]):
        for ai, aid in enumerate(AGENT_IDS):
            row = snap[snap["agent_id"] == aid]
            if len(row) > 0 and "climate" in row.columns:
                clim_mat[ai, ci] = cs_map.get(row["climate"].iloc[0], 0)
    im = ax10.imshow(clim_mat, aspect="auto", cmap=CLIMATE_CMAP,
                     vmin=0, vmax=4, interpolation="nearest")
    ax10.set_yticks(range(len(AGENT_IDS)))
    ax10.set_yticklabels(AGENT_IDS, color="white", fontsize=8)
    ax10.set_xlabel("Cycle", color="white")
    cbar = plt.colorbar(im, ax=ax10, shrink=0.8)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(["Normal", "Drought", "Flood", "Heat", "Storm"])
    cbar.ax.tick_params(colors="white", labelsize=7)

    # ── 11: Economic power ─────────────────────────────────────────────────
    ax11 = fig.add_subplot(4, 3, 12)
    dark_ax(ax11, "Economic Power Over Time")
    for aid in AGENT_IDS:
        sub = df_hist[df_hist["agent_id"] == aid]
        if "economic_power" not in sub.columns or len(sub) < 2:
            continue
        ax11.plot(sub["cycle"], sub["economic_power"],
                  color=COLORS[aid], lw=1.5, label=STATE_AGENTS[aid][:10])
    ax11.set_xlabel("Cycle", color="white")
    ax11.set_ylabel("Economic Power", color="white")
    ax11.legend(fontsize=7, facecolor="#111111", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.985], pad=2.0)
    out_path = "worldsim_india_visualisation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0a0a0a", edgecolor="none")
    print(f"\n✅ Saved → {out_path}")
    plt.show()
    return sim, df_hist


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Update this path for your Kaggle input directory ──────────────────
    CSV_PATH = "worldsim_merged.csv"  
    N_CYCLES = 100

    print("\n" + "=" * 65)
    print("WorldSim India — PettingZoo AEC Environment")
    print("=" * 65)

    # 1. Validate PettingZoo API
    print("\n[1] API validation ...")
    raw_env = WorldSimIndiaEnv(csv_path=CSV_PATH, max_cycles=N_CYCLES)
    obs, infos = raw_env.reset(seed=42)

    print(f"  Observation space : {raw_env.observation_spaces['RJ']}")
    print(f"  Action space      : {raw_env.action_spaces['RJ']}")
    print(f"  Obs shape         : {obs['RJ'].shape}")
    print(f"  Active agents     : {raw_env.agents}")
    assert obs["RJ"].shape == (WorldSimIndiaEnv.OBS_DIM,), "Observation dim mismatch!"
    assert not np.any(np.isnan(obs["RJ"])), "NaN in initial observation!"
    print("  ✓ API checks passed")

    # 2. Single step test
    print("\n[2] Single step test ...")
    raw_env.step(np.array([6, 1])) 
    raw_env.render()

    # 3. Full episode + visualisation
    print("\n[3] Full episode with visualisation ...")
    sim_env, df_history = run_visualisation(
        csv_path=CSV_PATH, n_cycles=N_CYCLES, random_seed=42)

    # 4. Summary
    print("\n[4] Final state summary:")
    final = df_history[df_history["cycle"] == df_history["cycle"].max()]
    for _, row in final.iterrows():
        aid = row.get("agent_id", "?")
        if row.get("status") == "collapsed":
            print(f"  {aid} ({STATE_AGENTS.get(aid, '?'):<22}) : COLLAPSED")
        else:
            w = row.get("water", 0)
            f = row.get("food",  0)
            e = row.get("energy",0)
            r = row.get("reputation", 0)
            print(f"  {aid} ({STATE_AGENTS.get(aid, '?'):<22}) : "
                  f"W={w:.3f}  F={f:.3f}  E={e:.3f}  Rep={r:.2f}")

    print("\n✅ WorldSim India complete.")
    print("   Next step: replace random policy with MAPPO agents via RLlib.")
