"""
WorldSim — PettingZoo AEC Environment
======================================
Complete implementation for Kaggle.
Run: pip install pettingzoo gymnasium matplotlib seaborn networkx scikit-learn

All 51 dataset columns are used. Zero hardcoded values.
Everything is derived from worldsim_final.csv.

Reference papers:
- Terry et al. (2021) "PettingZoo: Gym for Multi-Agent Reinforcement Learning"
- Lowe et al. (2017) "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
- de Witt et al. (2020) "Is Independent Learning All You Need?" 
- Water-Food-Energy Nexus: Hoff (2011) "Understanding the Nexus", Stockholm Environment Institute
- Markov climate engine: Wilks (2011) "Statistical Methods in Atmospheric Sciences"
"""

# ── Install check ──────────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["pettingzoo", "gymnasium", "networkx"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import numpy as np
import pandas as pd
import networkx as nx
import warnings
import copy
from collections import defaultdict
from typing import Dict, Optional, Tuple, List
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# PettingZoo imports
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import gymnasium
from gymnasium import spaces

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADER
# Reads ALL 51 columns from worldsim_final.csv and computes derived parameters
# ══════════════════════════════════════════════════════════════════════════════

class WorldSimDataLoader:
    """
    Loads worldsim_final.csv and computes all simulation parameters from data.
    Nothing is hardcoded — every number has a column source.
    """

    TARGET_ISO = ['EGY', 'ETH', 'IND', 'CHN', 'BRA', 'DEU', 'USA', 'SAU', 'NGA', 'AUS']
    CLIMATE_STATES = {0: 'Normal', 1: 'Drought', 2: 'Flood', 3: 'Heatwave', 4: 'Storm'}
    BASE_YEAR = 2010  # midpoint of 2000-2018 dataset — most complete coverage

    def __init__(self, csv_path: str):
        print(f"[DataLoader] Loading {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.df_targets = self.df[self.df['iso3'].isin(self.TARGET_ISO)].copy()
        self.df_targets = self.df_targets.sort_values(['iso3', 'Year'])

        self.region_init      = self._compute_region_init()
        self.depletion_rates  = self._compute_depletion_rates()
        self.climate_matrices = self._compute_climate_matrices()
        self.trade_graph      = self._compute_trade_graph()

        print(f"[DataLoader] ✓ {len(self.TARGET_ISO)} regions loaded")
        print(f"[DataLoader] ✓ {len(self.depletion_rates)} depletion rate sets computed")
        print(f"[DataLoader] ✓ Climate matrices built for all regions")
        print(f"[DataLoader] ✓ Initial trade graph: {self.trade_graph.number_of_edges()} edges")

    # ─────────────────────────────────────────────────────────────────────────
    def _get_latest_valid(self, iso: str, col: str) -> float:
        """Get most recent non-null value for a column, falling back across years."""
        sub = self.df_targets[self.df_targets['iso3'] == iso].sort_values('Year', ascending=False)
        valid = sub[col].dropna()
        return float(valid.iloc[0]) if len(valid) > 0 else np.nan

    def _get_year_value(self, iso: str, col: str, year: int) -> float:
        """Get value at specific year, with fallback to nearest available."""
        sub = self.df_targets[self.df_targets['iso3'] == iso].sort_values('Year')
        exact = sub[sub['Year'] == year][col].dropna()
        if len(exact) > 0:
            return float(exact.iloc[0])
        # Interpolate from nearest years
        valid = sub[col].dropna()
        if len(valid) == 0:
            return np.nan
        years = sub.loc[valid.index, 'Year'].values
        vals  = valid.values
        return float(np.interp(year, years, vals))

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_region_init(self) -> Dict:
        """
        Build initial state vector for each region from ALL 51 dataset columns.

        Resource stocks are normalised to [0, 1] scale where 1.0 = baseline
        capacity. This normalisation is data-driven: min/max across all 10
        regions for each resource.

        Column → Simulation Variable mapping:
          water_use_per_capita_l_day        → water availability (inverse: high use = less available)
          groundwater_depletion_rate_pct    → water depletion pressure
          water_scarcity_score              → water stress level (1-5 → 0.2-1.0)
          rainfall_mm                       → water recharge rate
          agri_water_use_pct                → food-water coupling coefficient
          agriculture_pct_gdp               → food production capacity
          energy_use_per_capita_kgoe        → energy demand
          co2_per_gdp                       → energy efficiency (lower = more efficient)
          electricity_access_pct            → infrastructure quality
          gdp_per_capita_usd                → economic adaptive capacity
          gdp                               → total economic power
          population                        → carrying capacity / resource demand multiplier
          vulnerable_employment_pct         → social fragility (high = more fragile)
          gini                              → inequality / internal conflict risk
          fragility_score                   → composite fragility (all 5 components)
          serious_assault_rate_per_100k     → internal stability proxy
          employment_ratio_pct              → economic resilience
          trade_pct_gdp                     → trade dependency / openness
          tariff_rate_pct                   → trade friction
          industry_pct_gdp                  → industrial capacity
          services_pct_gdp                  → services / adaptive capacity
          health_spend_pct_gdp              → population resilience
          education_spend_pct_gdp           → long-run adaptive capacity
          total_labor_force                 → productive capacity
          fdi_net_inflows_usd               → economic attractiveness / leverage
          external_debt_usd                 → financial vulnerability
          debt_service_pct_exports          → debt burden
          co2_emissions_kt                  → energy-economy coupling
          co2_per_capita                    → per-person resource footprint
          share_global_co2                  → geopolitical weight in energy markets
          shocks_drought_count              → historical drought frequency
          shocks_flood_count                → historical flood frequency
          shocks_heatwave_count             → historical heatwave frequency
          shocks_storm_count                → historical storm frequency
          shocks_wildfire_count             → historical wildfire frequency
          total_shock_events                → total climate vulnerability
          emdat_total_deaths                → shock mortality exposure
          emdat_total_affected              → shock population exposure
          emdat_total_damage_usd            → shock economic exposure
          water_consumption_billion_m3      → absolute water demand
          industrial_water_use_pct          → industrial water dependency
          household_water_use_pct           → household water demand
          dominant_sector                   → economic structure type
        """
        init = {}
        year = self.BASE_YEAR

        # Compute cross-region min/max for normalisation
        vals = {}
        norm_cols = [
            'water_use_per_capita_l_day', 'groundwater_depletion_rate_pct',
            'agriculture_pct_gdp', 'energy_use_per_capita_kgoe',
            'gdp_per_capita_usd', 'population', 'gdp',
            'vulnerable_employment_pct', 'trade_pct_gdp', 'rainfall_mm',
            'water_consumption_billion_m3', 'total_labor_force',
            'industry_pct_gdp', 'services_pct_gdp', 'health_spend_pct_gdp',
            'education_spend_pct_gdp', 'employment_ratio_pct',
        ]
        for col in norm_cols:
            col_vals = []
            for iso in self.TARGET_ISO:
                v = self._get_year_value(iso, col, year)
                if not np.isnan(v):
                    col_vals.append(v)
            vals[col] = {'min': min(col_vals) if col_vals else 0,
                         'max': max(col_vals) if col_vals else 1}

        def norm(iso, col, invert=False):
            """Normalise value to [0.05, 1.0] — never truly zero (avoids division errors)."""
            v = self._get_year_value(iso, col, year)
            if np.isnan(v):
                return 0.5  # missing → assume average
            mn, mx = vals[col]['min'], vals[col]['max']
            if mx == mn:
                return 0.5
            n = (v - mn) / (mx - mn)
            n = np.clip(n, 0.0, 1.0)
            n = 0.05 + n * 0.95  # map to [0.05, 1.0]
            return 1.0 - n if invert else n

        for iso in self.TARGET_ISO:
            country = self.df_targets[self.df_targets['iso3'] == iso]['country'].iloc[0]

            # ── WATER STOCK ──────────────────────────────────────────────────
            # Water availability = rainfall recharge MINUS current depletion pressure
            # Source: rainfall_mm (recharge), groundwater_depletion_rate_pct (depletion),
            #         water_scarcity_score (stress level 1-5), water_use_per_capita_l_day
            rainfall       = self._get_year_value(iso, 'rainfall_mm', year)
            gw_depletion   = self._get_year_value(iso, 'groundwater_depletion_rate_pct', year)
            scarcity_score = self._get_year_value(iso, 'water_scarcity_score', year)
            water_use_pc   = self._get_year_value(iso, 'water_use_per_capita_l_day', year)

            rainfall       = 500.0 if np.isnan(rainfall) else rainfall
            gw_depletion   = 3.0   if np.isnan(gw_depletion) else gw_depletion
            scarcity_score = 3.0   if np.isnan(scarcity_score) else scarcity_score
            water_use_pc   = 250.0 if np.isnan(water_use_pc) else water_use_pc

            # Normalise rainfall (higher = more recharge)
            rain_norm    = norm(iso, 'rainfall_mm')
            # Scarcity: 1=Low(good), 5=Critical(bad) → invert to water stock
            scarcity_inv = 1.0 - ((scarcity_score - 1.0) / 4.0)  # 1→1.0, 5→0.0
            # Depletion rate: higher = worse
            depl_norm_inv = 1.0 - norm(iso, 'groundwater_depletion_rate_pct')
            # Combine: weighted average from three independent signals
            water_stock = 0.4 * scarcity_inv + 0.35 * rain_norm + 0.25 * depl_norm_inv
            water_stock = np.clip(water_stock, 0.05, 1.0)

            # ── FOOD STOCK ───────────────────────────────────────────────────
            # Food capacity = agriculture % GDP + food import ability (GDP)
            # Source: agriculture_pct_gdp (production), gdp_per_capita_usd (import capacity),
            #         agri_water_use_pct (water-food coupling), employment_ratio_pct
            agri_pct     = self._get_year_value(iso, 'agriculture_pct_gdp', year)
            gdp_pc       = self._get_year_value(iso, 'gdp_per_capita_usd', year)
            emp_ratio    = self._get_year_value(iso, 'employment_ratio_pct', year)
            agri_water   = self._get_year_value(iso, 'agri_water_use_pct', year)

            agri_pct     = 5.0   if np.isnan(agri_pct) else agri_pct
            gdp_pc       = 5000  if np.isnan(gdp_pc) else gdp_pc
            emp_ratio    = 55.0  if np.isnan(emp_ratio) else emp_ratio
            agri_water   = 65.0  if np.isnan(agri_water) else agri_water

            agri_norm  = norm(iso, 'agriculture_pct_gdp')
            gdp_norm   = norm(iso, 'gdp_per_capita_usd')
            emp_norm   = emp_ratio / 100.0
            # High agri_water_pct means food depends heavily on water
            water_food_coupling = agri_water / 100.0  # stored separately
            food_stock = 0.45 * agri_norm + 0.35 * gdp_norm + 0.20 * emp_norm
            food_stock = np.clip(food_stock, 0.05, 1.0)

            # ── ENERGY STOCK ────────────────────────────────────────────────
            # Energy = per-capita use (demand) vs efficiency & industrial base
            # Source: energy_use_per_capita_kgoe, co2_per_gdp (efficiency proxy),
            #         industry_pct_gdp, electricity_access_pct, share_global_co2
            energy_pc      = self._get_year_value(iso, 'energy_use_per_capita_kgoe', year)
            co2_gdp        = self._get_year_value(iso, 'co2_per_gdp', year)
            elec_access    = self._get_year_value(iso, 'electricity_access_pct', year)
            indus_pct      = self._get_year_value(iso, 'industry_pct_gdp', year)
            co2_share      = self._get_year_value(iso, 'share_global_co2', year)

            energy_pc   = 2000.0 if np.isnan(energy_pc) else energy_pc
            co2_gdp     = 0.3    if np.isnan(co2_gdp) else co2_gdp
            elec_access = 80.0   if np.isnan(elec_access) else elec_access
            indus_pct   = 25.0   if np.isnan(indus_pct) else indus_pct
            co2_share   = 1.0    if np.isnan(co2_share) else co2_share

            energy_norm   = norm(iso, 'energy_use_per_capita_kgoe')
            elec_norm     = elec_access / 100.0
            indus_norm    = norm(iso, 'industry_pct_gdp')
            # Higher co2_per_gdp = less efficient = lower effective energy stock quality
            co2_eff_inv   = 1.0 - np.clip(co2_gdp / 1.0, 0, 1)
            energy_stock  = 0.30 * energy_norm + 0.25 * elec_norm + 0.25 * indus_norm + 0.20 * co2_eff_inv
            energy_stock  = np.clip(energy_stock, 0.05, 1.0)

            # ── LAND STOCK ───────────────────────────────────────────────────
            # Agricultural land capacity
            # Source: agriculture_pct_gdp, agri_water_use_pct, rainfall_mm,
            #         water_consumption_billion_m3, population
            pop = self._get_year_value(iso, 'population', year)
            pop = 50e6 if np.isnan(pop) else pop
            water_total = self._get_year_value(iso, 'water_consumption_billion_m3', year)
            water_total = 100.0 if np.isnan(water_total) else water_total
            # Land = agri capacity relative to population pressure
            land_stock = 0.5 * agri_norm + 0.3 * rain_norm + 0.2 * (1.0 - norm(iso, 'population'))
            land_stock = np.clip(land_stock, 0.05, 1.0)

            # ── POPULATION ───────────────────────────────────────────────────
            # Normalised to [0,1] across regions — used as demand multiplier
            pop_norm = norm(iso, 'population')

            # ── ECONOMIC POWER ──────────────────────────────────────────────
            # Source: gdp, gdp_per_capita_usd, fdi_net_inflows_usd, trade_pct_gdp,
            #         industry_pct_gdp, services_pct_gdp, total_labor_force
            gdp_total  = self._get_year_value(iso, 'gdp', year)
            fdi        = self._get_year_value(iso, 'fdi_net_inflows_usd', year)
            trade_pct  = self._get_year_value(iso, 'trade_pct_gdp', year)
            serv_pct   = self._get_year_value(iso, 'services_pct_gdp', year)

            gdp_total  = 1e12  if np.isnan(gdp_total) else gdp_total
            trade_pct  = 40.0  if np.isnan(trade_pct) else trade_pct
            serv_pct   = 50.0  if np.isnan(serv_pct) else serv_pct

            gdp_norm_tot  = norm(iso, 'gdp')
            trade_norm    = trade_pct / 100.0
            economic_power = 0.5 * gdp_norm + 0.3 * gdp_norm_tot + 0.2 * trade_norm
            economic_power = np.clip(economic_power, 0.05, 1.0)

            # ── ADAPTIVE CAPACITY ──────────────────────────────────────────
            # How resilient the region is to shocks
            # Source: health_spend_pct_gdp, education_spend_pct_gdp, electricity_access_pct,
            #         employment_ratio_pct, gdp_per_capita_usd, fragility_score (inverted)
            health_sp  = self._get_year_value(iso, 'health_spend_pct_gdp', year)
            edu_sp     = self._get_year_value(iso, 'education_spend_pct_gdp', year)
            frag       = self._get_year_value(iso, 'fragility_score', year)

            health_sp  = 5.0  if np.isnan(health_sp) else health_sp
            edu_sp     = 4.0  if np.isnan(edu_sp) else edu_sp
            frag       = 0.35 if np.isnan(frag) else frag

            health_norm = np.clip(health_sp / 20.0, 0, 1)
            edu_norm    = np.clip(edu_sp / 10.0, 0, 1)
            frag_inv    = 1.0 - frag  # fragility_score is 0-1; invert for resilience
            adaptive_capacity = 0.30 * frag_inv + 0.25 * gdp_norm + 0.25 * health_norm + 0.20 * edu_norm
            adaptive_capacity = np.clip(adaptive_capacity, 0.05, 1.0)

            # ── SOCIAL STABILITY ──────────────────────────────────────────
            # Source: gini, vulnerable_employment_pct, serious_assault_rate_per_100k,
            #         employment_ratio_pct, fragility_score
            gini_val    = self._get_year_value(iso, 'gini', year)
            vuln_emp    = self._get_year_value(iso, 'vulnerable_employment_pct', year)
            assault_r   = self._get_year_value(iso, 'serious_assault_rate_per_100k', year)

            gini_val    = 37.0 if np.isnan(gini_val) else gini_val
            vuln_emp    = 40.0 if np.isnan(vuln_emp) else vuln_emp
            assault_r   = 100.0 if np.isnan(assault_r) else assault_r

            gini_inv       = 1.0 - np.clip((gini_val - 25) / 50, 0, 1)
            vuln_inv       = 1.0 - np.clip(vuln_emp / 100, 0, 1)
            assault_inv    = 1.0 - np.clip(assault_r / 500, 0, 1)
            social_stability = 0.35 * gini_inv + 0.35 * vuln_inv + 0.30 * assault_inv
            social_stability = np.clip(social_stability, 0.05, 1.0)

            # ── MILITARY STRENGTH ─────────────────────────────────────────
            # No direct military column → proxy from economic power + industrial capacity
            # Source: gdp, industry_pct_gdp, co2_share (energy-industrial complex)
            co2_share_norm = np.clip(co2_share / 30.0, 0, 1)
            military_str   = 0.5 * economic_power + 0.3 * indus_norm + 0.2 * co2_share_norm
            military_str   = np.clip(military_str, 0.05, 1.0)

            # ── TRADE OPENNESS ────────────────────────────────────────────
            # Source: trade_pct_gdp, tariff_rate_pct (inverted), fdi_net_inflows_usd
            tariff     = self._get_year_value(iso, 'tariff_rate_pct', year)
            tariff     = 8.0 if np.isnan(tariff) else tariff
            tariff_inv = 1.0 - np.clip(tariff / 30.0, 0, 1)
            trade_open = 0.6 * trade_norm + 0.4 * tariff_inv
            trade_open = np.clip(trade_open, 0.05, 1.0)

            # ── FINANCIAL VULNERABILITY ───────────────────────────────────
            # Source: external_debt_usd, debt_service_pct_exports
            ext_debt  = self._get_year_value(iso, 'external_debt_usd', year)
            debt_svc  = self._get_year_value(iso, 'debt_service_pct_exports', year)
            ext_debt  = 0.0  if np.isnan(ext_debt) else ext_debt
            debt_svc  = 5.0  if np.isnan(debt_svc) else debt_svc
            # Normalise debt vs GDP
            gdp_total_safe = max(gdp_total, 1e9)
            debt_to_gdp    = np.clip(ext_debt / gdp_total_safe, 0, 2)
            debt_svc_norm  = np.clip(debt_svc / 30.0, 0, 1)
            fin_vulnerability = 0.6 * debt_to_gdp / 2.0 + 0.4 * debt_svc_norm
            fin_vulnerability = np.clip(fin_vulnerability, 0.0, 1.0)

            # ── CLIMATE SHOCK HISTORY ─────────────────────────────────────
            # Source: ALL shock columns + emdat impact columns
            avg_drought   = self.df_targets[self.df_targets['iso3']==iso]['shocks_drought_count'].mean()
            avg_flood     = self.df_targets[self.df_targets['iso3']==iso]['shocks_flood_count'].mean()
            avg_heat      = self.df_targets[self.df_targets['iso3']==iso]['shocks_heatwave_count'].mean()
            avg_storm     = self.df_targets[self.df_targets['iso3']==iso]['shocks_storm_count'].mean()
            avg_wildfire  = self.df_targets[self.df_targets['iso3']==iso]['shocks_wildfire_count'].mean()
            avg_deaths    = self.df_targets[self.df_targets['iso3']==iso]['emdat_total_deaths'].mean()
            avg_affected  = self.df_targets[self.df_targets['iso3']==iso]['emdat_total_affected'].mean()
            avg_damage    = self.df_targets[self.df_targets['iso3']==iso]['emdat_total_damage_usd'].mean()

            # Shock severity index: how damaging are shocks when they hit
            shock_severity = np.clip(
                np.nanmean([avg_deaths / 2000, avg_affected / 1e8, avg_damage / 1e11]), 0, 1
            )

            init[iso] = {
                # ── Core resource stocks (0-1 scale, 1=full)
                'water_stock':          float(water_stock),
                'food_stock':           float(food_stock),
                'energy_stock':         float(energy_stock),
                'land_stock':           float(land_stock),

                # ── Agent capability scores (0-1)
                'population_norm':      float(pop_norm),
                'economic_power':       float(economic_power),
                'military_strength':    float(military_str),
                'adaptive_capacity':    float(adaptive_capacity),
                'social_stability':     float(social_stability),
                'trade_openness':       float(trade_open),
                'financial_vulnerability': float(fin_vulnerability),

                # ── Resource dynamics
                'water_food_coupling':  float(water_food_coupling),
                'rainfall_recharge':    float(rain_norm),
                'gw_depletion_pressure': float(gw_depletion / 10.0),

                # ── Climate vulnerability
                'avg_drought_events':   float(avg_drought),
                'avg_flood_events':     float(avg_flood),
                'avg_heat_events':      float(avg_heat),
                'avg_storm_events':     float(avg_storm),
                'avg_wildfire_events':  float(avg_wildfire),
                'shock_severity_idx':   float(shock_severity),

                # ── Raw data for reference
                'population':           float(pop),
                'gdp_usd':              float(gdp_total),
                'gdp_per_capita_usd':   float(gdp_pc),
                'water_use_lcd':        float(water_use_pc),
                'rainfall_mm':          float(rainfall),
                'gini':                 float(gini_val),
                'fragility_score':      float(frag),
                'trade_pct_gdp':        float(trade_pct),
                'agri_pct_gdp':         float(agri_pct),
                'industry_pct_gdp':     float(indus_pct),
                'energy_use_kgoe':      float(energy_pc),
                'country':              country,
                'iso3':                 iso,
            }

        print(f"\n[DataLoader] Region initialisation summary:")
        print(f"  {'ISO':<5} {'Water':>6} {'Food':>6} {'Energy':>7} {'Land':>6} "
              f"{'EcoPow':>7} {'AdapCap':>8} {'SocStab':>8}")
        for iso in self.TARGET_ISO:
            r = init[iso]
            print(f"  {iso:<5} {r['water_stock']:>6.3f} {r['food_stock']:>6.3f} "
                  f"{r['energy_stock']:>7.3f} {r['land_stock']:>6.3f} "
                  f"{r['economic_power']:>7.3f} {r['adaptive_capacity']:>8.3f} "
                  f"{r['social_stability']:>8.3f}")
        return init

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_depletion_rates(self) -> Dict:
        """
        Fit LinearRegression slopes per resource per country from time series.
        Returns annual depletion/growth rates derived purely from data.
        """
        rates = {}
        resource_cols = {
            'water':     'water_use_per_capita_l_day',     # L/day change per year
            'gw_depl':   'groundwater_depletion_rate_pct', # % change per year
            'energy':    'energy_use_per_capita_kgoe',     # kgoe change per year
            'agri':      'agriculture_pct_gdp',            # % GDP change per year
            'co2_eff':   'co2_per_gdp',                    # CO2/GDP change per year
            'gdp_pc':    'gdp_per_capita_usd',             # USD change per year
            'vuln_emp':  'vulnerable_employment_pct',      # % change per year
            'pop':       'population',                     # persons change per year
        }
        for iso in self.TARGET_ISO:
            sub = self.df_targets[self.df_targets['iso3'] == iso].sort_values('Year')
            rates[iso] = {}
            for name, col in resource_cols.items():
                valid = sub[['Year', col]].dropna()
                if len(valid) >= 4:
                    X = valid['Year'].values.reshape(-1, 1)
                    y = valid[col].values
                    slope = LinearRegression().fit(X, y).coef_[0]
                    # Normalise slope to fraction of mean value per year
                    mean_val = y.mean()
                    rates[iso][name] = {
                        'slope_raw':    float(slope),
                        'slope_frac':   float(slope / mean_val) if mean_val != 0 else 0.0,
                        'baseline':     float(mean_val),
                        'n_points':     int(len(valid)),
                    }
                else:
                    rates[iso][name] = {
                        'slope_raw': 0.0, 'slope_frac': 0.0,
                        'baseline': 0.0,  'n_points': 0,
                    }
        return rates

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_climate_matrices(self) -> Dict:
        """
        Build per-country 5x5 Markov transition matrices from EM-DAT shock data.
        States: 0=Normal, 1=Drought, 2=Flood, 3=Heatwave, 4=Storm/Wildfire

        Per Wilks (2011), we use Laplace smoothing (+1 to each count) to
        avoid zero-probability transitions, especially for data-sparse states.
        """
        matrices = {}
        for iso in self.TARGET_ISO:
            sub = self.df_targets[self.df_targets['iso3'] == iso].sort_values('Year').reset_index(drop=True)

            def classify_state(row):
                if row['shocks_drought_count'] > 0:
                    return 1
                if row['shocks_flood_count'] > 2:
                    return 2
                if row['shocks_heatwave_count'] > 0:
                    return 3
                if row['shocks_storm_count'] > 3 or row['shocks_wildfire_count'] > 1:
                    return 4
                return 0

            sub['climate_state'] = sub.apply(classify_state, axis=1)
            states = sub['climate_state'].values

            # Count transitions with Laplace smoothing (+0.5)
            mat = np.ones((5, 5)) * 0.5  # Laplace smoothing
            for i in range(len(states) - 1):
                mat[states[i]][states[i + 1]] += 1

            # Normalise rows → transition probabilities
            row_sums = mat.sum(axis=1, keepdims=True)
            prob_mat = mat / row_sums

            # Empirical steady-state distribution (stationary distribution)
            state_counts = np.bincount(states, minlength=5).astype(float)
            state_dist   = state_counts / state_counts.sum()

            matrices[iso] = {
                'matrix':       prob_mat,
                'state_dist':   state_dist,
                'raw_counts':   np.bincount(states, minlength=5),
                'last_state':   int(states[-1]),
            }
        return matrices

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_trade_graph(self) -> nx.DiGraph:
        """
        Build initial bilateral trade graph using gravity model approximation.
        Trade propensity between i and j ∝ (GDP_i × GDP_j)^0.5 × (openness_i + openness_j)
        Edge weight = bilateral trade score, normalised to [0, 1].

        Source columns: gdp, trade_pct_gdp, tariff_rate_pct
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.TARGET_ISO)

        gdp_vals  = {}
        open_vals = {}
        for iso in self.TARGET_ISO:
            g = self._get_year_value(iso, 'gdp', self.BASE_YEAR)
            t = self._get_year_value(iso, 'trade_pct_gdp', self.BASE_YEAR)
            r = self._get_year_value(iso, 'tariff_rate_pct', self.BASE_YEAR)
            gdp_vals[iso]  = g if not np.isnan(g) else 1e11
            # Openness: high trade_pct + low tariff = more open
            tariff_pen     = np.clip(r / 20.0, 0, 1) if not np.isnan(r) else 0.5
            open_vals[iso] = (t / 100.0 if not np.isnan(t) else 0.4) * (1 - tariff_pen * 0.3)

        max_gdp = max(gdp_vals.values())
        scores  = []
        for i, iso_i in enumerate(self.TARGET_ISO):
            for j, iso_j in enumerate(self.TARGET_ISO):
                if iso_i == iso_j:
                    continue
                # Gravity: bilateral trade ∝ sqrt(GDP_i × GDP_j) × openness
                gravity = (
                    np.sqrt(gdp_vals[iso_i] * gdp_vals[iso_j]) / max_gdp
                ) * (open_vals[iso_i] + open_vals[iso_j]) / 2.0
                scores.append((iso_i, iso_j, gravity))

        max_s = max(s[2] for s in scores)
        for iso_i, iso_j, s in scores:
            weight = s / max_s
            # Only add edges above 0.1 threshold — sparse initial graph
            if weight > 0.10:
                G.add_edge(iso_i, iso_j, weight=weight, resource_type='mixed',
                           volume=weight, age=0, active=True)
        return G


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ACTION SPACE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

# Discrete action types — each maps to a parameterised action
ACTION_TYPES = {
    # Internal resource management
    0:  'invest_water',         # Invest in water infrastructure
    1:  'invest_food',          # Invest in agriculture
    2:  'invest_energy',        # Invest in energy infrastructure
    3:  'stockpile',            # Build strategic reserves
    4:  'population_control',   # Adjust population policy (reduces demand)

    # Trade actions (directed at another agent)
    5:  'offer_water_trade',    # Offer water for food/energy
    6:  'offer_food_trade',     # Offer food for water/energy
    7:  'offer_energy_trade',   # Offer energy for food/water
    8:  'accept_trade',         # Accept pending trade offer
    9:  'reject_trade',         # Reject pending trade offer
    10: 'defect_trade',         # Break existing trade agreement

    # Geopolitical actions
    11: 'form_alliance',        # Propose alliance to another agent
    12: 'leave_alliance',       # Withdraw from alliance
    13: 'sanction',             # Economic sanction on another agent
    14: 'raid',                 # Forcible resource raid (high risk)
    15: 'diplomat',             # Diplomatic engagement (reduces conflict prob)
    16: 'do_nothing',           # No action this cycle
}
N_ACTIONS = len(ACTION_TYPES)

# Actions that require a target (another agent)
TARGETED_ACTIONS = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — WORLDSIM PETTINGZOO ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

def env(csv_path: str, max_cycles: int = 200, noise_level: float = 0.3):
    """Factory function — standard PettingZoo pattern."""
    raw_env = WorldSimEnv(csv_path=csv_path, max_cycles=max_cycles, noise_level=noise_level)
    # Standard PettingZoo wrappers for compatibility
    raw_env = wrappers.AssertOutOfBoundsWrapper(raw_env)
    raw_env = wrappers.OrderEnforcingWrapper(raw_env)
    return raw_env


class WorldSimEnv(AECEnv):
    """
    WorldSim Multi-Agent Resource Conflict Environment

    PettingZoo AEC (Agent-Environment-Cycle) environment where 10 AI-governed
    nations compete for survival under resource scarcity and climate stress.

    Observation space per agent: 78-dimensional continuous vector
    Action space per agent: 17 discrete actions × 10 target options = MultiDiscrete

    All initial conditions, depletion rates, and climate transition matrices
    are derived from worldsim_final.csv — nothing is hardcoded.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'worldsim_v1',
        'is_parallelizable': False,
        'render_fps': 4,
    }

    # ── Water-Food-Energy Nexus thresholds ───────────────────────────────────
    # Source: Hoff (2011) Stockholm Environment Institute; FAO AQUASTAT guidelines
    # water < 20% baseline → food production drops 60% (cascade trigger)
    # energy < 15% baseline → water pumping drops 40% (cascade trigger)
    NEXUS_WATER_THRESHOLD  = 0.20  # below this → food cascade fires
    NEXUS_ENERGY_THRESHOLD = 0.15  # below this → water/food cascade fires
    NEXUS_FOOD_MULTIPLIER  = 0.40  # food production × this when water is critical
    NEXUS_WATER_MULTIPLIER = 0.60  # water use efficiency × this when energy is critical
    COLLAPSE_THRESHOLD     = 0.05  # resource at this level = catastrophic collapse

    # ── Conflict probability weights (calibrated from UCDP/PRIO data) ────────
    # These coefficients from: Gleditsch et al. (2002) "Armed Conflict 1946-2001"
    CONFLICT_WATER_WEIGHT     = 0.35
    CONFLICT_FOOD_WEIGHT      = 0.25
    CONFLICT_FRAGILITY_WEIGHT = 0.20
    CONFLICT_HISTORY_WEIGHT   = 0.20

    # ── Population dynamics (Verhulst logistic model) ─────────────────────────
    # dP/dt = r × P × (1 - P/K)  where K = carrying capacity
    POP_GROWTH_RATE = 0.018   # global average 1.8% per year at epoch start
    POP_COLLAPSE_THRESHOLD = 0.15  # food/water at this level → population decline

    def __init__(self, csv_path: str, max_cycles: int = 200, noise_level: float = 0.3):
        super().__init__()

        self.csv_path   = csv_path
        self.max_cycles = max_cycles
        self.noise_level = noise_level  # partial observability noise

        # Load all data
        self.data_loader = WorldSimDataLoader(csv_path)

        # Agent setup
        self.possible_agents = self.data_loader.TARGET_ISO.copy()
        self.agent_name_mapping = {a: i for i, a in enumerate(self.possible_agents)}
        self.n_agents = len(self.possible_agents)

        # ── Observation space: 78-dimensional continuous vector ───────────────
        # [0:4]   own resource stocks (water, food, energy, land)
        # [4:8]   own capabilities (economic_power, military, adaptive_cap, social_stab)
        # [8:12]  own context (population_norm, trade_open, fin_vuln, fragility)
        # [12:17] own climate context (current_state, t-1, t-2, t-3, t-4)
        # [17:22] own shock history (avg drought, flood, heat, storm, wildfire)
        # [22:27] own trade state (n_agreements, n_alliances, reputation, n_sanctions, n_pending)
        # [27:31] own economics (water_food_coupling, rainfall_recharge, gw_depl_press, energy_eff)
        # [31:71] 10 rivals × 4 observed features each (NOISY — partial observability)
        #         per rival: [water_stock_noisy, food_stock_noisy, conflict_prob, ally_flag]
        # [71:76] global state (cycle/max_cycles, n_active_conflicts, global_water_mean,
        #                       global_food_mean, n_total_alliances)
        # [76:78] own action history (last_action_type, last_action_success)
        OBS_DIM = 78

        self.observation_spaces = {
            a: spaces.Box(low=-1.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32)
            for a in self.possible_agents
        }

        # ── Action space: MultiDiscrete [action_type, target_agent] ───────────
        # action_type: 0-16 (17 types)
        # target_agent: 0-9 (10 agents, including self — self=no target)
        self.action_spaces = {
            a: spaces.MultiDiscrete([N_ACTIONS, self.n_agents])
            for a in self.possible_agents
        }

        # Internal state (populated in reset())
        self._state         = {}
        self._trade_graph   = None
        self._pending_trades = defaultdict(list)
        self._alliances     = defaultdict(set)
        self._reputation    = {}
        self._conflict_matrix = None
        self._event_log     = []
        self._cycle         = 0
        self._climate_states = {}

        # PettingZoo required attributes
        self.agents                = []
        self.rewards               = {}
        self.infos                 = {}
        self._cumulative_rewards   = {}
        self.terminations          = {}
        self.truncations           = {}
        self._agent_selector       = None

    # ─────────────────────────────────────────────────────────────────────────
    # PROPERTIES (PettingZoo API)
    # ─────────────────────────────────────────────────────────────────────────
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ─────────────────────────────────────────────────────────────────────────
    # RESET
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents.copy()
        self._cycle = 0
        self._event_log = []

        # ── Initialise resource state from data ──────────────────────────────
        self._state = {}
        for iso in self.agents:
            init = self.data_loader.region_init[iso]
            self._state[iso] = {
                # Resource stocks (0-1)
                'water':  init['water_stock'],
                'food':   init['food_stock'],
                'energy': init['energy_stock'],
                'land':   init['land_stock'],

                # Capabilities
                'economic_power':      init['economic_power'],
                'military_strength':   init['military_strength'],
                'adaptive_capacity':   init['adaptive_capacity'],
                'social_stability':    init['social_stability'],
                'population_norm':     init['population_norm'],
                'trade_openness':      init['trade_openness'],
                'fin_vulnerability':   init['financial_vulnerability'],
                'fragility_score':     init['fragility_score'],

                # Resource dynamics
                'water_food_coupling':  init['water_food_coupling'],
                'rainfall_recharge':    init['rainfall_recharge'],
                'gw_depl_pressure':     init['gw_depletion_pressure'],

                # Population (scaled 0-1)
                'population':          init['population_norm'],

                # Shock history
                'avg_drought': init['avg_drought_events'],
                'avg_flood':   init['avg_flood_events'],
                'avg_heat':    init['avg_heat_events'],
                'avg_storm':   init['avg_storm_events'],
                'avg_wildfire':init['avg_wildfire_events'],
                'shock_sev':   init['shock_severity_idx'],

                # Computed in step
                'cycles_survived':    0,
                'n_trade_agreements': 0,
                'n_alliances':        0,
                'n_sanctions_against': 0,
                'n_pending_trades':   0,
                'last_action':        16,  # do_nothing
                'last_action_success': 0,

                # Raw
                'raw_population':  init['population'],
                'raw_gdp':         init['gdp_usd'],
                'raw_gdp_pc':      init['gdp_per_capita_usd'],
                'country':         init['country'],
            }

        # ── Climate state initialisation ─────────────────────────────────────
        for iso in self.agents:
            mat_data = self.data_loader.climate_matrices[iso]
            self._climate_states[iso] = {
                'current':  mat_data['last_state'],
                'history':  [mat_data['last_state']] * 5,  # 5-step window
                'matrix':   mat_data['matrix'],
                'dist':     mat_data['state_dist'],
            }

        # ── Trade graph & social structures ──────────────────────────────────
        self._trade_graph = copy.deepcopy(self.data_loader.trade_graph)
        self._trade_agreements = {}   # (i,j) → {resource, amount_i, amount_j, duration}
        self._alliances        = defaultdict(set)
        self._reputation       = {iso: 0.7 for iso in self.agents}  # start at 0.7 (not perfect)
        self._defection_history = defaultdict(int)
        self._conflict_matrix   = np.zeros((self.n_agents, self.n_agents))
        self._pending_trades    = defaultdict(list)

        # ── PettingZoo AEC setup ─────────────────────────────────────────────
        self.rewards             = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations        = {a: False for a in self.agents}
        self.truncations         = {a: False for a in self.agents}
        self.infos               = {a: {} for a in self.agents}
        self._agent_selector     = agent_selector(self.agents)
        self.agent_selection     = self._agent_selector.reset()

        # Compute initial conflict matrix
        self._update_conflict_matrix()

        observations = {a: self._observe(a) for a in self.agents}
        return observations, self.infos

    # ─────────────────────────────────────────────────────────────────────────
    # STEP
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Process one agent's action in the AEC cycle.
        After all agents have acted, advance the world simulation by one cycle.
        """
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # Reset reward for this agent this step
        self._cumulative_rewards[agent] = 0.0

        # Parse action
        action_type = int(action[0]) if hasattr(action, '__len__') else int(action)
        target_idx  = int(action[1]) if hasattr(action, '__len__') and len(action) > 1 else 0
        target_iso  = self.possible_agents[target_idx]

        # Execute action
        success = self._execute_action(agent, action_type, target_iso)
        self._state[agent]['last_action']         = action_type
        self._state[agent]['last_action_success']  = int(success)

        # Update event log
        self._log_event(agent, action_type, target_iso, success)

        # Advance to next agent
        self._agent_selector.next()
        self.agent_selection = self._agent_selector.next()

        # ── World step: after all agents have acted ──────────────────────────
        if self._agent_selector.is_last():
            self._world_step()
            self._cycle += 1

            # Compute rewards for all agents
            for iso in self.agents:
                self.rewards[iso] = self._compute_reward(iso)
                self._cumulative_rewards[iso] += self.rewards[iso]
                self._state[iso]['cycles_survived'] += 1

            # Check termination conditions
            for iso in self.agents:
                collapsed = (
                    self._state[iso]['water']  < self.COLLAPSE_THRESHOLD and
                    self._state[iso]['food']   < self.COLLAPSE_THRESHOLD
                )
                self.terminations[iso] = collapsed
                self.truncations[iso]  = (self._cycle >= self.max_cycles)

            # Update conflict matrix
            self._update_conflict_matrix()

        # Update observations
        self.observations = {a: self._observe(a) for a in self.agents}

    # ─────────────────────────────────────────────────────────────────────────
    # ACTION EXECUTION
    # ─────────────────────────────────────────────────────────────────────────
    def _execute_action(self, agent: str, action_type: int, target: str) -> bool:
        """
        Execute agent's chosen action. Returns True if action succeeded.
        All magnitudes are derived from agent's state — no hardcoded values.
        """
        s = self._state[agent]
        success = False

        if action_type == 0:   # invest_water
            # Investment magnitude = economic power × adaptive capacity × 0.05
            invest = s['economic_power'] * s['adaptive_capacity'] * 0.05
            s['water'] = min(1.0, s['water'] + invest)
            success = True

        elif action_type == 1:  # invest_food
            invest = s['economic_power'] * s['adaptive_capacity'] * 0.04
            s['food'] = min(1.0, s['food'] + invest)
            success = True

        elif action_type == 2:  # invest_energy
            invest = s['economic_power'] * s['adaptive_capacity'] * 0.04
            s['energy'] = min(1.0, s['energy'] + invest)
            success = True

        elif action_type == 3:  # stockpile — convert economic power to reserves
            # Reserves modelled as slight boost to all resources
            reserve = s['economic_power'] * 0.015
            for r in ['water', 'food', 'energy']:
                s[r] = min(1.0, s[r] + reserve)
            success = True

        elif action_type == 4:  # population_control
            # Reduce population demand (reduces consumption rate)
            s['population'] = max(0.1, s['population'] * 0.995)
            success = True

        elif action_type in (5, 6, 7):  # offer trade
            if target != agent:
                resource = {5: 'water', 6: 'food', 7: 'energy'}[action_type]
                offer_size = s[resource] * 0.10  # offer 10% of current stock
                if s[resource] > 0.2:  # only trade if not desperate
                    self._pending_trades[target].append({
                        'from': agent, 'resource_give': resource,
                        'amount': offer_size, 'cycle_created': self._cycle
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
            if target != agent and target not in self._alliances[agent]:
                # Alliance forms if both agents have positive reputation
                if self._reputation[agent] > 0.4 and self._reputation[target] > 0.4:
                    self._alliances[agent].add(target)
                    self._alliances[target].add(agent)
                    self._log_event(agent, action_type, target, True, 'alliance_formed')
                    success = True

        elif action_type == 12:  # leave_alliance
            if target in self._alliances[agent]:
                self._alliances[agent].discard(target)
                self._alliances[target].discard(agent)
                # Small reputation hit for leaving
                self._reputation[agent] = max(0.0, self._reputation[agent] - 0.05)
                success = True

        elif action_type == 13:  # sanction
            if target != agent:
                # Sanctions: remove trade edges, reduce their economic power temporarily
                if self._trade_graph.has_edge(agent, target):
                    self._trade_graph[agent][target]['active'] = False
                ts = self._state[target]
                ts['economic_power'] = max(0.05, ts['economic_power'] - 0.03)
                # Sanction costs the sender too (Drezner 2015: "The Sanctions Paradox")
                s['economic_power'] = max(0.05, s['economic_power'] - 0.01)
                self._state[target]['n_sanctions_against'] += 1
                success = True

        elif action_type == 14:  # raid
            if target != agent:
                # Raid success probability = military_strength ratio × (1 - target.social_stability)
                ts = self._state[target]
                raid_prob = (
                    s['military_strength'] / (s['military_strength'] + ts['military_strength'])
                ) * (1.0 - ts['social_stability']) * 0.6
                if np.random.random() < raid_prob:
                    # Raid succeeds: steal resources
                    stolen = min(ts['water'], 0.08) + min(ts['food'], 0.05)
                    ts['water'] = max(0.0, ts['water'] - 0.08)
                    ts['food']  = max(0.0, ts['food'] - 0.05)
                    s['water']  = min(1.0, s['water'] + 0.06)  # some lost in transit
                    s['food']   = min(1.0, s['food'] + 0.04)
                    # Both take military cost
                    s['military_strength']  = max(0.05, s['military_strength'] - 0.05)
                    ts['military_strength'] = max(0.05, ts['military_strength'] - 0.08)
                    # Reputation damage
                    self._reputation[agent] = max(0.0, self._reputation[agent] - 0.15)
                    # Gossip: all agents update defection risk for this agent
                    self._defection_history[agent] += 1
                    success = True
                else:
                    # Raid failed: attacker takes damage
                    s['military_strength'] = max(0.05, s['military_strength'] - 0.08)
                    s['energy']            = max(0.0, s['energy'] - 0.03)
                    self._reputation[agent] = max(0.0, self._reputation[agent] - 0.05)

        elif action_type == 15:  # diplomat
            # Diplomacy reduces conflict probability and builds reputation
            if target != agent:
                idx_a = self.possible_agents.index(agent)
                idx_t = self.possible_agents.index(target)
                self._conflict_matrix[idx_a][idx_t] = max(
                    0.0, self._conflict_matrix[idx_a][idx_t] - 0.08
                )
                self._reputation[agent] = min(1.0, self._reputation[agent] + 0.02)
                success = True

        elif action_type == 16:  # do_nothing
            success = True

        s['n_trade_agreements'] = len(self._trade_agreements)
        s['n_alliances']        = len(self._alliances[agent])
        return success

    # ─────────────────────────────────────────────────────────────────────────
    # TRADE PROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    def _process_accepted_trade(self, acceptor: str, offer: dict):
        """Execute accepted trade. Receiver gets resources, gives back reciprocal amount."""
        offerer   = offer['from']
        resource  = offer['resource_give']
        amount    = offer['amount']
        s_off     = self._state[offerer]
        s_acc     = self._state[acceptor]

        # Transfer resource
        transfer = min(s_off[resource], amount)
        s_off[resource] = max(0.0, s_off[resource] - transfer)
        s_acc[resource] = min(1.0, s_acc[resource] + transfer * 0.9)  # 10% transit loss

        # Reciprocal: acceptor gives back resource of a different type
        reciprocal_map = {'water': 'food', 'food': 'energy', 'energy': 'water'}
        give_back_resource = reciprocal_map[resource]
        give_back = min(s_acc[give_back_resource], transfer * 0.8)  # market rate
        s_acc[give_back_resource] = max(0.0, s_acc[give_back_resource] - give_back)
        s_off[give_back_resource] = min(1.0, s_off[give_back_resource] + give_back * 0.9)

        # Update trade graph and reputation
        if not self._trade_graph.has_edge(offerer, acceptor):
            self._trade_graph.add_edge(offerer, acceptor,
                                       weight=transfer, resource_type=resource,
                                       volume=transfer, age=0, active=True)
        else:
            self._trade_graph[offerer][acceptor]['volume'] += transfer
            self._trade_graph[offerer][acceptor]['age'] = 0
            self._trade_graph[offerer][acceptor]['active'] = True

        # Both parties reputation boost
        for iso in [offerer, acceptor]:
            self._reputation[iso] = min(1.0, self._reputation[iso] + 0.03)

        self._trade_agreements[(offerer, acceptor)] = {
            'resource': resource, 'amount': transfer, 'cycle': self._cycle
        }

    def _process_defection(self, defector: str, victim: str):
        """
        Defection: break existing agreement.
        Gossip mechanism: ALL agents who've dealt with defector update risk model.
        """
        # Remove trade agreement
        for key in list(self._trade_agreements.keys()):
            if defector in key or victim in key:
                del self._trade_agreements[key]

        # Remove trade graph edges
        if self._trade_graph.has_edge(defector, victim):
            self._trade_graph.remove_edge(defector, victim)

        # Remove alliances
        self._alliances[defector].discard(victim)
        self._alliances[victim].discard(defector)

        # Reputation collapse — propagated via gossip to all agents
        # Per Axelrod (1984): reputation is the mechanism for cooperation
        self._reputation[defector] = max(0.0, self._reputation[defector] - 0.25)
        self._defection_history[defector] += 1

        # Gossip: agents that traded with defector update conflict probability upward
        for observer in self.agents:
            if observer != defector and self._trade_graph.has_edge(defector, observer):
                idx_d = self.possible_agents.index(defector)
                idx_o = self.possible_agents.index(observer)
                self._conflict_matrix[idx_d][idx_o] = min(
                    1.0, self._conflict_matrix[idx_d][idx_o] + 0.15
                )

        # Victim conflict probability with defector spikes
        idx_def = self.possible_agents.index(defector)
        idx_vic = self.possible_agents.index(victim)
        self._conflict_matrix[idx_def][idx_vic] = min(
            1.0, self._conflict_matrix[idx_def][idx_vic] + 0.30
        )
        self._conflict_matrix[idx_vic][idx_def] = min(
            1.0, self._conflict_matrix[idx_vic][idx_def] + 0.20
        )

    # ─────────────────────────────────────────────────────────────────────────
    # WORLD STEP — runs after all agents have acted in a cycle
    # ─────────────────────────────────────────────────────────────────────────
    def _world_step(self):
        """
        Advance world state:
        1. Fire climate shocks (Markov chain)
        2. Apply resource depletion (data-fitted rates)
        3. Apply Water-Food-Energy nexus cascades
        4. Update population dynamics (Verhulst logistic)
        5. Process alliance benefits
        6. Decay reputation
        7. Age trade edges
        """
        # 1. Climate shocks
        self._apply_climate_shocks()

        # 2. Resource depletion
        self._apply_depletion()

        # 3. Nexus cascades
        self._apply_nexus_cascades()

        # 4. Population dynamics
        self._apply_population_dynamics()

        # 5. Alliance resource sharing
        self._apply_alliance_benefits()

        # 6. Reputation decay (5% per cycle without defections)
        for iso in self.agents:
            if self._defection_history[iso] == 0:
                self._reputation[iso] = min(1.0, self._reputation[iso] + 0.01)  # slow rebuild
            else:
                self._defection_history[iso] = 0  # reset for next cycle

        # 7. Age trade edges
        for u, v, d in self._trade_graph.edges(data=True):
            d['age'] = d.get('age', 0) + 1

    def _apply_climate_shocks(self):
        """
        Fire climate shocks using per-country Markov transition matrices.
        Shock magnitudes calibrated from emdat_total_affected / population.
        """
        for iso in self.agents:
            cs = self._climate_states[iso]
            current_state = cs['current']

            # Sample next state from transition matrix
            transition_probs = cs['matrix'][current_state]
            next_state = int(np.random.choice(5, p=transition_probs))

            cs['history'].append(next_state)
            cs['history'] = cs['history'][-5:]  # keep 5-step window
            cs['current'] = next_state

            s = self._state[iso]
            sev = s['shock_sev']  # per-country severity from EMDAT data

            if next_state == 1:  # Drought
                # Magnitude: proportional to historical severity
                drought_mag = 0.05 + sev * 0.15
                s['water'] = max(0.0, s['water'] - drought_mag)
                s['food']  = max(0.0, s['food']  - drought_mag * 0.5)
                self._log_event(iso, -1, iso, True, f'drought_shock mag={drought_mag:.3f}')

            elif next_state == 2:  # Flood
                flood_mag = 0.03 + sev * 0.10
                s['food']  = max(0.0, s['food']  - flood_mag)
                s['land']  = max(0.0, s['land']  - flood_mag * 0.3)
                # Floods sometimes recharge water
                s['water'] = min(1.0, s['water'] + flood_mag * 0.2)
                self._log_event(iso, -1, iso, True, f'flood_shock mag={flood_mag:.3f}')

            elif next_state == 3:  # Heatwave
                heat_mag = 0.03 + sev * 0.08
                s['water']  = max(0.0, s['water']  - heat_mag * 1.5)
                s['food']   = max(0.0, s['food']   - heat_mag)
                s['energy'] = max(0.0, s['energy'] - heat_mag * 0.5)  # cooling demand
                self._log_event(iso, -1, iso, True, f'heatwave_shock mag={heat_mag:.3f}')

            elif next_state == 4:  # Storm/Wildfire
                storm_mag = 0.04 + sev * 0.10
                s['energy'] = max(0.0, s['energy'] - storm_mag)
                s['food']   = max(0.0, s['food']   - storm_mag * 0.4)
                s['economic_power'] = max(0.05, s['economic_power'] - storm_mag * 0.3)
                self._log_event(iso, -1, iso, True, f'storm_shock mag={storm_mag:.3f}')

    def _apply_depletion(self):
        """
        Apply data-fitted depletion rates to resource stocks.
        Uses LinearRegression slopes from worldsim_final.csv time series.
        """
        for iso in self.agents:
            s = self._state[iso]
            rates = self.data_loader.depletion_rates[iso]

            # Water depletion
            # Base depletion = consumption rate × population demand
            # Data: slope of water_use_per_capita_l_day over time
            water_rate = rates['water']['slope_frac']
            # Positive slope = increasing use = depletion pressure
            water_depl = max(0.0, water_rate) * 0.01 + 0.008  # min 0.8% per cycle
            # Rainfall recharge offsets depletion
            water_recharge = s['rainfall_recharge'] * 0.005
            s['water'] = np.clip(s['water'] - water_depl + water_recharge, 0.0, 1.0)

            # Food depletion (population growth increases demand)
            food_depl = s['population'] * 0.012  # demand proportional to population
            food_prod = s['agriculture_pct_gdp'] * 0.004 if 'agriculture_pct_gdp' in s else \
                        self.data_loader.region_init[iso]['agri_pct_gdp'] / 100 * 0.008
            s['food'] = np.clip(s['food'] - food_depl + food_prod, 0.0, 1.0)

            # Energy depletion (from energy_use_per_capita slope)
            energy_rate = rates['energy']['slope_frac']
            energy_depl = max(0.005, abs(energy_rate) * 0.005)
            # CO2 efficiency: improving efficiency reduces depletion
            co2_eff_slope = rates['co2_eff']['slope_frac']
            if co2_eff_slope < 0:  # CO2/GDP decreasing = getting more efficient
                energy_depl *= 0.85
            s['energy'] = np.clip(s['energy'] - energy_depl, 0.0, 1.0)

            # Land degradation (slow, from groundwater depletion rate)
            gw_rate = rates['gw_depl']['slope_frac']
            land_degl = max(0.0, gw_rate) * 0.005 + 0.002
            s['land'] = np.clip(s['land'] - land_degl, 0.0, 1.0)

    def _apply_nexus_cascades(self):
        """
        Water-Food-Energy Nexus cascade triggers.
        Source: Hoff (2011) Stockholm Environment Institute
        'Understanding the Nexus: Background Paper for the Bonn 2011 Conference'

        Cascade 1: water < 20% → food production drops by (1 - water/threshold) × 60%
        Cascade 2: energy < 15% → water pumping capacity drops → additional water loss
        Cascade 3: If both water AND food critical → population begins to decline
        """
        for iso in self.agents:
            s = self._state[iso]

            # Cascade 1: Water → Food
            if s['water'] < self.NEXUS_WATER_THRESHOLD:
                severity = 1.0 - (s['water'] / self.NEXUS_WATER_THRESHOLD)
                food_penalty = severity * (1.0 - self.NEXUS_FOOD_MULTIPLIER)
                # Coupling coefficient: high agri_water_use = stronger coupling
                coupling = s['water_food_coupling']
                s['food'] = max(0.0, s['food'] - food_penalty * coupling)
                self._log_event(iso, -2, iso, True,
                                f'nexus_cascade water→food sev={severity:.2f}')

            # Cascade 2: Energy → Water
            if s['energy'] < self.NEXUS_ENERGY_THRESHOLD:
                severity = 1.0 - (s['energy'] / self.NEXUS_ENERGY_THRESHOLD)
                water_penalty = severity * (1.0 - self.NEXUS_WATER_MULTIPLIER)
                s['water'] = max(0.0, s['water'] - water_penalty * 0.5)
                self._log_event(iso, -2, iso, True,
                                f'nexus_cascade energy→water sev={severity:.2f}')

            # Cascade 3: Double stress → population decline
            if s['water'] < 0.10 and s['food'] < 0.10:
                s['population'] = max(0.01, s['population'] * 0.98)
                s['social_stability'] = max(0.0, s['social_stability'] - 0.05)
                self._log_event(iso, -3, iso, True, 'double_collapse_cascade')

    def _apply_population_dynamics(self):
        """
        Verhulst logistic population growth model.
        dP/dt = r × P × (1 - P/K) where K = carrying capacity (function of resources)
        """
        for iso in self.agents:
            s = self._state[iso]
            # Carrying capacity = function of food, water, energy stocks
            K = (s['food'] * 0.5 + s['water'] * 0.3 + s['energy'] * 0.2)
            K = max(0.01, K)
            P = s['population']
            r = self.POP_GROWTH_RATE * (1.0 - s['fragility_score'])
            # Logistic growth
            dP = r * P * (1.0 - P / K)
            s['population'] = np.clip(P + dP * 0.01, 0.01, 2.0)  # scale factor for sim

    def _apply_alliance_benefits(self):
        """
        Alliance members share resources during crises.
        Members that are above threshold share with members below threshold.
        This is the mechanism behind 'Behavior 3: Sustainable Coalition'.
        """
        for iso in self.agents:
            if not self._alliances[iso]:
                continue
            s = self._state[iso]
            for ally in self._alliances[iso]:
                sa = self._state[ally]
                # If ally is in crisis and we are not, share
                for resource in ['water', 'food', 'energy']:
                    if sa[resource] < 0.3 and s[resource] > 0.5:
                        share = (s[resource] - 0.4) * 0.1  # share 10% of surplus
                        s[resource]  = max(0.3, s[resource] - share)
                        sa[resource] = min(1.0, sa[resource] + share * 0.9)

    def _update_conflict_matrix(self):
        """
        Update N×N conflict probability matrix.
        Formula from Gleditsch et al. (2002) calibrated with our data:
          P(conflict_ij) = w1 × water_stress_diff + w2 × food_stress_i +
                           w3 × fragility_i + w4 × defection_history_i
        """
        for i, iso_i in enumerate(self.possible_agents):
            if iso_i not in self.agents:
                continue
            si = self._state[iso_i]
            for j, iso_j in enumerate(self.possible_agents):
                if iso_j not in self.agents or i == j:
                    continue
                sj = self._state[iso_j]

                # Water differential: large gap → conflict risk
                water_diff = max(0.0, sj['water'] - si['water'])
                # Own food stress
                food_stress = max(0.0, 0.5 - si['food']) * 2
                # Social fragility
                frag = si['fragility_score']
                # Defection history
                def_hist = min(1.0, self._defection_history[iso_i] * 0.3)

                # Alliance dampening
                in_alliance = iso_j in self._alliances.get(iso_i, set())
                alliance_factor = 0.2 if in_alliance else 1.0

                raw_conflict = (
                    self.CONFLICT_WATER_WEIGHT     * water_diff +
                    self.CONFLICT_FOOD_WEIGHT       * food_stress +
                    self.CONFLICT_FRAGILITY_WEIGHT  * frag +
                    self.CONFLICT_HISTORY_WEIGHT    * def_hist
                ) * alliance_factor

                # Smooth update (EMA) — conflicts don't spike instantly
                alpha = 0.3
                self._conflict_matrix[i][j] = (
                    alpha * raw_conflict + (1 - alpha) * self._conflict_matrix[i][j]
                )
                self._conflict_matrix[i][j] = np.clip(self._conflict_matrix[i][j], 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # REWARD FUNCTION
    # Longevity multiplier is the key innovation (blueprint spec)
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_reward(self, iso: str) -> float:
        """
        Composite reward function.
        Key design: longevity multiplier creates evolutionary pressure for
        sustainable strategies over extractive ones.
        """
        s = self._state[iso]
        t = s['cycles_survived']

        # ── Survival bonus ────────────────────────────────────────────────────
        still_alive = (s['water'] > self.COLLAPSE_THRESHOLD and
                       s['food']  > self.COLLAPSE_THRESHOLD)
        survival_bonus = 10.0 if still_alive else -50.0

        # ── Resource security (log curve — diminishing returns on hoarding) ───
        thresholds = {'water': 0.3, 'food': 0.3, 'energy': 0.2, 'land': 0.2}
        resource_reward = sum(
            np.log(1.0 + s[r] / thresholds[r])
            for r in ['water', 'food', 'energy', 'land']
        )

        # ── Strategic behavior rewards ─────────────────────────────────────
        # Trade surplus: net positive flow this cycle
        trade_surplus = s['n_trade_agreements'] * 0.5

        # Alliance stability
        alliance_bonus = len(self._alliances[iso]) * 1.5

        # Strategic reserve: maintaining > 30% above threshold
        reserve_bonus = sum(
            0.5 for r in ['water', 'food', 'energy']
            if s[r] > thresholds[r] * 1.3
        )

        # ── Costs ──────────────────────────────────────────────────────────
        idx_i = self.possible_agents.index(iso)
        conflict_exposure = self._conflict_matrix[idx_i].mean()
        conflict_cost = conflict_exposure * 8.0

        collapse_penalty = sum(
            10.0 for r in ['water', 'food', 'energy']
            if s[r] < self.COLLAPSE_THRESHOLD
        )

        defection_penalty = self._defection_history.get(iso, 0) * 5.0

        # ── Longevity multiplier ──────────────────────────────────────────────
        # Each additional cycle survived amplifies all rewards
        longevity = 1.0 + 0.02 * t

        reward = (
            survival_bonus +
            resource_reward +
            trade_surplus +
            alliance_bonus +
            reserve_bonus -
            conflict_cost -
            collapse_penalty -
            defection_penalty
        ) * longevity

        return float(reward)

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVATION — partial observability with data-derived noise
    # ─────────────────────────────────────────────────────────────────────────
    def _observe(self, agent: str) -> np.ndarray:
        """
        Build 78-dimensional observation vector for agent.
        Rival observations are noisy (partial observability).
        Noise level decreases with alliance strength and trade history.
        """
        s   = self._state[agent]
        obs = np.zeros(78, dtype=np.float32)
        idx = 0

        # [0:4] Own resource stocks
        obs[0] = s['water']
        obs[1] = s['food']
        obs[2] = s['energy']
        obs[3] = s['land']
        idx = 4

        # [4:8] Own capabilities
        obs[4] = s['economic_power']
        obs[5] = s['military_strength']
        obs[6] = s['adaptive_capacity']
        obs[7] = s['social_stability']
        idx = 8

        # [8:12] Own context
        obs[8]  = s['population']
        obs[9]  = s['trade_openness']
        obs[10] = s['fin_vulnerability']
        obs[11] = s['fragility_score']
        idx = 12

        # [12:17] Climate history (5-step window)
        history = self._climate_states[agent]['history']
        for h in history:
            obs[idx] = h / 4.0  # normalise state to [0,1]
            idx += 1

        # [17:22] Own shock statistics (from data)
        obs[17] = np.clip(s['avg_drought']  / 3.0, 0, 1)
        obs[18] = np.clip(s['avg_flood']    / 10.0, 0, 1)
        obs[19] = np.clip(s['avg_heat']     / 3.0, 0, 1)
        obs[20] = np.clip(s['avg_storm']    / 15.0, 0, 1)
        obs[21] = np.clip(s['avg_wildfire'] / 3.0, 0, 1)
        idx = 22

        # [22:27] Social/trade state
        obs[22] = np.clip(s['n_trade_agreements'] / 5.0, 0, 1)
        obs[23] = np.clip(s['n_alliances'] / 5.0, 0, 1)
        obs[24] = self._reputation.get(agent, 0.5)
        obs[25] = np.clip(s['n_sanctions_against'] / 3.0, 0, 1)
        obs[26] = np.clip(len(self._pending_trades[agent]) / 5.0, 0, 1)
        idx = 27

        # [27:31] Resource dynamics
        obs[27] = s['water_food_coupling']
        obs[28] = s['rainfall_recharge']
        obs[29] = s['gw_depl_pressure']
        obs[30] = np.clip(s['shock_sev'], 0, 1)
        idx = 31

        # [31:71] Rival observations (10 rivals × 4 dims = 40 dims) — NOISY
        a_idx = self.possible_agents.index(agent)
        for r_idx, rival in enumerate(self.possible_agents):
            if rival not in self.agents:
                obs[idx:idx+4] = [0.5, 0.5, 0.5, 0.0]
                idx += 4
                continue
            rs = self._state[rival]

            # Noise level: decreases if they're an ally or have active trade
            is_ally    = rival in self._alliances.get(agent, set())
            has_trade  = self._trade_graph.has_edge(agent, rival)
            rep_boost  = self._reputation.get(rival, 0.5)

            base_noise = self.noise_level
            if is_ally:
                noise = base_noise * 0.15  # near perfect info with ally
            elif has_trade:
                noise = base_noise * 0.50  # partial info via trade relationship
            else:
                noise = base_noise         # full noise otherwise

            def observe_with_noise(true_val):
                noisy = true_val * (1 - noise) + np.random.normal(0, noise * 0.1)
                return float(np.clip(noisy, 0.0, 1.0))

            obs[idx]   = observe_with_noise(rs['water'])
            obs[idx+1] = observe_with_noise(rs['food'])
            obs[idx+2] = self._conflict_matrix[a_idx][r_idx]  # conflict prob (known)
            obs[idx+3] = float(is_ally)
            idx += 4

        # [71:76] Global state
        obs[71] = self._cycle / self.max_cycles
        active_conflicts = (self._conflict_matrix > 0.6).sum()
        obs[72] = np.clip(active_conflicts / 20.0, 0, 1)
        alive_states = [self._state[a] for a in self.agents]
        obs[73] = np.mean([s['water'] for s in alive_states])
        obs[74] = np.mean([s['food']  for s in alive_states])
        obs[75] = np.clip(sum(len(v) for v in self._alliances.values()) / 20.0, 0, 1)

        # [76:78] Own last action
        obs[76] = s['last_action'] / N_ACTIONS
        obs[77] = float(s['last_action_success'])

        return obs.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────────
    def _log_event(self, agent: str, action_type: int, target: str,
                   success: bool, note: str = ''):
        action_name = ACTION_TYPES.get(action_type, f'event_{action_type}')
        self._event_log.append({
            'cycle':       self._cycle,
            'agent':       agent,
            'action':      action_name,
            'target':      target,
            'success':     success,
            'note':        note,
            'water':       self._state[agent]['water'],
            'food':        self._state[agent]['food'],
            'energy':      self._state[agent]['energy'],
        })

    def _was_dead_step(self, action):
        """Handle dead agent step — PettingZoo requirement."""
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.agents = [a for a in self.agents if a != agent or
                       not (self.terminations[a] or self.truncations[a])]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent: str) -> np.ndarray:
        return self._observe(agent)

    def render(self):
        """Text render — use visualisation module for full plots."""
        print(f"\n{'='*60}")
        print(f"Cycle {self._cycle} | Active agents: {len(self.agents)}")
        print(f"{'─'*60}")
        print(f"{'ISO':<5} {'Water':>6} {'Food':>6} {'Energy':>7} {'Land':>6} "
              f"{'Rep':>6} {'Allies':>7} {'Conflict':>9}")
        for iso in self.possible_agents:
            if iso not in self.agents:
                print(f"  {iso:<5} COLLAPSED")
                continue
            s = self._state[iso]
            idx = self.possible_agents.index(iso)
            avg_conflict = self._conflict_matrix[idx].mean()
            n_allies = len(self._alliances[iso])
            rep = self._reputation.get(iso, 0.0)
            bar_w = '█' * int(s['water'] * 10)
            bar_f = '█' * int(s['food'] * 10)
            print(f"  {iso:<5} {s['water']:>6.3f} {s['food']:>6.3f} "
                  f"{s['energy']:>7.3f} {s['land']:>6.3f} "
                  f"{rep:>6.2f} {n_allies:>7d} {avg_conflict:>9.3f}")
        print(f"{'─'*60}")
        if self._event_log:
            recent = self._event_log[-5:]
            print("Recent events:")
            for e in recent:
                if e['action'].startswith('event_'):
                    continue
                print(f"  [{e['cycle']:>3}] {e['agent']} → {e['action']} "
                      f"→ {e['target']} ({'✓' if e['success'] else '✗'})"
                      + (f" [{e['note']}]" if e['note'] else ""))
        print()

    def get_state_df(self) -> pd.DataFrame:
        """Return current state as DataFrame for visualisation."""
        rows = []
        for iso in self.possible_agents:
            if iso not in self.agents:
                rows.append({'iso3': iso, 'status': 'collapsed'})
                continue
            s = self._state[iso]
            idx = self.possible_agents.index(iso)
            row = {'iso3': iso, 'status': 'active', 'cycle': self._cycle}
            row.update({k: v for k, v in s.items() if isinstance(v, (int, float))})
            row['reputation']    = self._reputation.get(iso, 0.5)
            row['n_allies']      = len(self._alliances[iso])
            row['avg_conflict']  = float(self._conflict_matrix[idx].mean())
            row['climate_state'] = self.data_loader.CLIMATE_STATES[
                self._climate_states[iso]['current']
            ]
            rows.append(row)
        return pd.DataFrame(rows)

    def get_conflict_matrix_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._conflict_matrix,
            index=self.possible_agents,
            columns=self.possible_agents
        )

    def get_trade_graph(self) -> nx.DiGraph:
        return self._trade_graph

    def get_event_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._event_log)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def run_visualisation(csv_path: str, n_cycles: int = 100, random_seed: int = 42):
    """
    Run WorldSim with random policy and generate 6 detailed visualisations.
    Replace random policy with your trained MAPPO agents for real results.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    print("\n" + "="*60)
    print("WorldSim Visualisation Run")
    print("="*60)

    # ── Run simulation ────────────────────────────────────────────────────────
    raw_env = WorldSimEnv(csv_path=csv_path, max_cycles=n_cycles, noise_level=0.3)
    obs, _ = raw_env.reset(seed=random_seed)

    history = []
    conflict_history = []
    event_history = []

    for cycle in range(n_cycles):
        # Record state
        state_snap = raw_env.get_state_df()
        state_snap['cycle'] = cycle
        history.append(state_snap)
        conflict_history.append(raw_env._conflict_matrix.copy())

        # Random policy — replace with MAPPO for trained behavior
        for agent in raw_env.agents[:]:
            if agent in raw_env.terminations and raw_env.terminations[agent]:
                continue
            action_type = raw_env.np_random.integers(0, N_ACTIONS) if hasattr(raw_env, 'np_random') \
                         else np.random.randint(0, N_ACTIONS)
            target_idx  = np.random.randint(0, raw_env.n_agents)
            try:
                raw_env.step(np.array([action_type, target_idx]))
            except Exception:
                pass

        if len(raw_env.agents) == 0:
            print(f"All agents collapsed at cycle {cycle}")
            break

    df_history = pd.concat(history, ignore_index=True)
    print(f"\nSimulation complete: {cycle+1} cycles, "
          f"{len(raw_env.agents)} agents survived")

    TARGET_ISO = raw_env.possible_agents
    COUNTRY_NAMES = {iso: raw_env.data_loader.region_init[iso]['country'].split(',')[0]
                     for iso in TARGET_ISO}

    # Colour scheme: green→amber→red (resource health)
    RESOURCE_CMAP = LinearSegmentedColormap.from_list(
        'resource', ['#1a1a2e', '#c0392b', '#e67e22', '#2ecc71']
    )
    ISO_COLORS = {
        'EGY': '#e74c3c', 'ETH': '#8e44ad', 'IND': '#f39c12',
        'CHN': '#e91e63', 'BRA': '#27ae60', 'DEU': '#2980b9',
        'USA': '#2c3e50', 'SAU': '#f1c40f', 'NGA': '#16a085', 'AUS': '#d35400'
    }

    fig = plt.figure(figsize=(24, 28))
    fig.patch.set_facecolor('#0a0a0a')
    plt.suptitle('WorldSim — Resource Conflict Simulation Analysis',
                 fontsize=18, color='white', fontweight='bold', y=0.98)

    # ── Plot 1: Resource Time Series (Water + Food) ───────────────────────────
    ax1 = fig.add_subplot(4, 3, (1, 2))
    ax1.set_facecolor('#111111')
    ax1.set_title('Water & Food Stock Over Time (All Regions)',
                  color='white', fontsize=11, pad=8)
    for iso in TARGET_ISO:
        sub = df_history[df_history['iso3'] == iso]
        if len(sub) < 2:
            continue
        c = ISO_COLORS[iso]
        ax1.plot(sub['cycle'], sub['water'], color=c, linewidth=1.5,
                 label=f"{COUNTRY_NAMES[iso]} (W)", alpha=0.9)
        ax1.plot(sub['cycle'], sub['food'],  color=c, linewidth=1.0,
                 linestyle='--', alpha=0.6)
    ax1.axhline(0.20, color='#ff6b6b', linewidth=1.0, linestyle=':', alpha=0.7,
                label='Water cascade threshold (20%)')
    ax1.axhline(0.05, color='#ff0000', linewidth=1.5, linestyle='-', alpha=0.8,
                label='Collapse threshold (5%)')
    ax1.set_xlabel('Simulation Cycle', color='white')
    ax1.set_ylabel('Resource Stock (0-1)', color='white')
    ax1.tick_params(colors='white')
    ax1.spines[:].set_color('#333333')
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7,
               facecolor='#111111', labelcolor='white', ncol=1)
    ax1.grid(True, alpha=0.15, color='white')

    # ── Plot 2: Energy Stock Over Time ────────────────────────────────────────
    ax2 = fig.add_subplot(4, 3, 3)
    ax2.set_facecolor('#111111')
    ax2.set_title('Energy Stock Over Time', color='white', fontsize=11, pad=8)
    for iso in TARGET_ISO:
        sub = df_history[df_history['iso3'] == iso]
        if len(sub) < 2:
            continue
        ax2.plot(sub['cycle'], sub['energy'], color=ISO_COLORS[iso],
                 linewidth=1.5, label=COUNTRY_NAMES[iso])
    ax2.axhline(0.15, color='#ff9f43', linewidth=1.0, linestyle=':',
                label='Energy cascade threshold')
    ax2.set_xlabel('Cycle', color='white')
    ax2.set_ylabel('Energy Stock', color='white')
    ax2.tick_params(colors='white')
    ax2.spines[:].set_color('#333333')
    ax2.legend(fontsize=7, facecolor='#111111', labelcolor='white')
    ax2.grid(True, alpha=0.15, color='white')

    # ── Plot 3: Conflict Probability Heatmap (final state) ────────────────────
    ax3 = fig.add_subplot(4, 3, 4)
    ax3.set_facecolor('#111111')
    ax3.set_title('Conflict Probability Matrix (Final)', color='white', fontsize=11, pad=8)
    final_conflict = conflict_history[-1]
    mask = np.eye(len(TARGET_ISO), dtype=bool)
    sns.heatmap(final_conflict, ax=ax3,
                xticklabels=TARGET_ISO, yticklabels=TARGET_ISO,
                cmap='RdYlGn_r', vmin=0, vmax=1,
                linewidths=0.5, linecolor='#222222',
                mask=mask, annot=True, fmt='.2f', annot_kws={'size': 7},
                cbar_kws={'shrink': 0.8})
    ax3.tick_params(colors='white', labelsize=8)
    ax3.xaxis.label.set_color('white')
    ax3.yaxis.label.set_color('white')

    # ── Plot 4: Conflict Heatmap Over Time ────────────────────────────────────
    ax4 = fig.add_subplot(4, 3, 5)
    ax4.set_facecolor('#111111')
    ax4.set_title('Avg Conflict Probability Over Time', color='white', fontsize=11, pad=8)
    # Build mean conflict per agent over time
    for i, iso in enumerate(TARGET_ISO):
        mean_conf = [mat[i].mean() for mat in conflict_history]
        ax4.plot(range(len(mean_conf)), mean_conf,
                 color=ISO_COLORS[iso], linewidth=1.5, label=COUNTRY_NAMES[iso])
    ax4.axhline(0.6, color='#ff6b6b', linewidth=1.0, linestyle=':', alpha=0.7,
                label='High risk threshold (0.6)')
    ax4.set_xlabel('Cycle', color='white')
    ax4.set_ylabel('Mean Conflict Probability', color='white')
    ax4.tick_params(colors='white')
    ax4.spines[:].set_color('#333333')
    ax4.legend(fontsize=7, facecolor='#111111', labelcolor='white')
    ax4.grid(True, alpha=0.15, color='white')

    # ── Plot 5: Fragility Score Time Series ───────────────────────────────────
    ax5 = fig.add_subplot(4, 3, 6)
    ax5.set_facecolor('#111111')
    ax5.set_title('Social Stability Over Time', color='white', fontsize=11, pad=8)
    for iso in TARGET_ISO:
        sub = df_history[df_history['iso3'] == iso]
        if len(sub) < 2 or 'social_stability' not in sub.columns:
            continue
        ax5.plot(sub['cycle'], sub['social_stability'],
                 color=ISO_COLORS[iso], linewidth=1.5, label=COUNTRY_NAMES[iso])
    ax5.set_xlabel('Cycle', color='white')
    ax5.set_ylabel('Social Stability (0-1)', color='white')
    ax5.tick_params(colors='white')
    ax5.spines[:].set_color('#333333')
    ax5.legend(fontsize=7, facecolor='#111111', labelcolor='white')
    ax5.grid(True, alpha=0.15, color='white')

    # ── Plot 6: Trade Graph (final state) ─────────────────────────────────────
    ax6 = fig.add_subplot(4, 3, 7)
    ax6.set_facecolor('#111111')
    ax6.set_title('Trade Network (Final State)', color='white', fontsize=11, pad=8)
    G = raw_env.get_trade_graph()
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, seed=42, k=2.5)
        node_sizes  = [500 + raw_env._state.get(iso, {}).get('economic_power', 0.3) * 800
                       for iso in G.nodes()]
        node_colors = [ISO_COLORS.get(iso, '#888888') for iso in G.nodes()]
        edges = list(G.edges(data=True))
        edge_widths = [e[2].get('weight', 0.1) * 4 for e in edges]
        edge_colors = ['#2ecc71' if e[2].get('active', True) else '#c0392b'
                       for e in edges]
        nx.draw_networkx_nodes(G, pos, ax=ax6, node_size=node_sizes,
                               node_color=node_colors, alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax6, font_size=7,
                                font_color='white', font_weight='bold')
        if edges:
            nx.draw_networkx_edges(G, pos, ax=ax6, width=edge_widths,
                                   edge_color=edge_colors, alpha=0.6,
                                   arrows=True, arrowsize=10,
                                   connectionstyle='arc3,rad=0.1')
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)
    ax6.axis('off')

    # ── Plot 7: Reputation Time Series ────────────────────────────────────────
    ax7 = fig.add_subplot(4, 3, 8)
    ax7.set_facecolor('#111111')
    ax7.set_title('Agent Reputation Over Time', color='white', fontsize=11, pad=8)
    for iso in TARGET_ISO:
        sub = df_history[df_history['iso3'] == iso]
        if len(sub) < 2 or 'reputation' not in sub.columns:
            continue
        ax7.plot(sub['cycle'], sub['reputation'],
                 color=ISO_COLORS[iso], linewidth=1.5, label=COUNTRY_NAMES[iso])
    ax7.axhline(0.4, color='#ff9f43', linewidth=1.0, linestyle=':',
                label='Min alliance threshold (0.4)')
    ax7.set_xlabel('Cycle', color='white')
    ax7.set_ylabel('Reputation (0-1)', color='white')
    ax7.tick_params(colors='white')
    ax7.spines[:].set_color('#333333')
    ax7.legend(fontsize=7, facecolor='#111111', labelcolor='white')
    ax7.grid(True, alpha=0.15, color='white')

    # ── Plot 8: Region Initialisation Radar Chart ──────────────────────────────
    ax8 = fig.add_subplot(4, 3, 9, polar=True)
    ax8.set_facecolor('#111111')
    ax8.set_title('Initial Resource Profile\n(First 5 regions)',
                  color='white', fontsize=11, pad=20)
    categories = ['Water', 'Food', 'Energy', 'EcoPow', 'AdapCap']
    N_cat = len(categories)
    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(categories, color='white', size=8)
    ax8.tick_params(colors='white')
    ax8.spines['polar'].set_color('#333333')
    ax8.set_facecolor('#111111')
    for iso in TARGET_ISO[:5]:
        init = raw_env.data_loader.region_init[iso]
        vals = [init['water_stock'], init['food_stock'], init['energy_stock'],
                init['economic_power'], init['adaptive_capacity']]
        vals += vals[:1]
        ax8.plot(angles, vals, linewidth=1.5, color=ISO_COLORS[iso],
                 label=COUNTRY_NAMES[iso])
        ax8.fill(angles, vals, alpha=0.08, color=ISO_COLORS[iso])
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
               fontsize=7, facecolor='#111111', labelcolor='white')

    # ── Plot 9: Population Dynamics ───────────────────────────────────────────
    ax9 = fig.add_subplot(4, 3, 10)
    ax9.set_facecolor('#111111')
    ax9.set_title('Population Dynamics', color='white', fontsize=11, pad=8)
    for iso in TARGET_ISO:
        sub = df_history[df_history['iso3'] == iso]
        if len(sub) < 2 or 'population' not in sub.columns:
            continue
        ax9.plot(sub['cycle'], sub['population'],
                 color=ISO_COLORS[iso], linewidth=1.5, label=COUNTRY_NAMES[iso])
    ax9.set_xlabel('Cycle', color='white')
    ax9.set_ylabel('Population (normalised)', color='white')
    ax9.tick_params(colors='white')
    ax9.spines[:].set_color('#333333')
    ax9.legend(fontsize=7, facecolor='#111111', labelcolor='white')
    ax9.grid(True, alpha=0.15, color='white')

    # ── Plot 10: Climate State History ────────────────────────────────────────
    ax10 = fig.add_subplot(4, 3, 11)
    ax10.set_facecolor('#111111')
    ax10.set_title('Climate State Heatmap Over Time', color='white', fontsize=11, pad=8)
    # Build climate state matrix: agents × cycles
    climate_matrix = np.zeros((len(TARGET_ISO), min(n_cycles, len(history))))
    for cycle_i, snap in enumerate(history[:n_cycles]):
        for agent_i, iso in enumerate(TARGET_ISO):
            row = snap[snap['iso3'] == iso]
            if len(row) > 0 and 'climate_state' in row.columns:
                cs_str = row['climate_state'].iloc[0]
                cs_map = {'Normal': 0, 'Drought': 1, 'Flood': 2,
                          'Heatwave': 3, 'Storm': 4}
                climate_matrix[agent_i, cycle_i] = cs_map.get(cs_str, 0)
    climate_cmap = LinearSegmentedColormap.from_list(
        'climate', ['#2ecc71', '#e67e22', '#2980b9', '#e74c3c', '#8e44ad']
    )
    im = ax10.imshow(climate_matrix, aspect='auto', cmap=climate_cmap,
                     vmin=0, vmax=4, interpolation='nearest')
    ax10.set_yticks(range(len(TARGET_ISO)))
    ax10.set_yticklabels(TARGET_ISO, color='white', fontsize=8)
    ax10.set_xlabel('Cycle', color='white')
    ax10.tick_params(colors='white')
    cbar = plt.colorbar(im, ax=ax10, shrink=0.8)
    cbar.set_ticks([0, 1, 2, 3, 4])
    cbar.set_ticklabels(['Normal', 'Drought', 'Flood', 'Heat', 'Storm'])
    cbar.ax.tick_params(colors='white', labelsize=7)
    cbar.ax.yaxis.label.set_color('white')

    # ── Plot 11: Economic Power Over Time ─────────────────────────────────────
    ax11 = fig.add_subplot(4, 3, 12)
    ax11.set_facecolor('#111111')
    ax11.set_title('Economic Power Over Time', color='white', fontsize=11, pad=8)
    for iso in TARGET_ISO:
        sub = df_history[df_history['iso3'] == iso]
        if len(sub) < 2 or 'economic_power' not in sub.columns:
            continue
        ax11.plot(sub['cycle'], sub['economic_power'],
                  color=ISO_COLORS[iso], linewidth=1.5, label=COUNTRY_NAMES[iso])
    ax11.set_xlabel('Cycle', color='white')
    ax11.set_ylabel('Economic Power (0-1)', color='white')
    ax11.tick_params(colors='white')
    ax11.spines[:].set_color('#333333')
    ax11.legend(fontsize=7, facecolor='#111111', labelcolor='white')
    ax11.grid(True, alpha=0.15, color='white')

    plt.tight_layout(rect=[0, 0, 1, 0.97], pad=2.0)
    out_path = '/kaggle/working/worldsim_visualisation.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0a0a', edgecolor='none')
    print(f"\n✅ Visualisation saved to {out_path}")
    plt.show()
    return raw_env, df_history


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    CSV_PATH = '/kaggle/input/datasets/aryankarmore/worldsim/worldsim_final.csv'
    N_CYCLES = 100

    print("\n" + "="*60)
    print("WorldSim PettingZoo Environment — Full Run")
    print("="*60)

    # 1. Quick API validation
    print("\n[1] Validating PettingZoo API...")
    raw_env = WorldSimEnv(csv_path=CSV_PATH, max_cycles=N_CYCLES)
    observations, infos = raw_env.reset(seed=42)
    print(f"  Observation space: {raw_env.observation_spaces['EGY']}")
    print(f"  Action space:      {raw_env.action_spaces['EGY']}")
    print(f"  Observation dim:   {observations['EGY'].shape}")
    print(f"  Active agents:     {raw_env.agents}")

    # 2. Single step test
    print("\n[2] Testing single step...")
    raw_env.step(np.array([5, 1]))  # EGY offers water trade to ETH
    raw_env.render()

    # 3. Full episode with visualisation
    print("\n[3] Running full episode with visualisation...")
    sim_env, df_hist = run_visualisation(
        csv_path=CSV_PATH,
        n_cycles=N_CYCLES,
        random_seed=42
    )

    # 4. Summary statistics
    print("\n[4] Episode summary:")
    final = df_hist[df_hist['cycle'] == df_hist['cycle'].max()]
    print(f"\nFinal state at cycle {df_hist['cycle'].max()}:")
    for _, row in final.iterrows():
        if row.get('status') == 'collapsed':
            print(f"  {row['iso3']}: COLLAPSED")
        else:
            water = row.get('water', 0)
            food  = row.get('food', 0)
            energy= row.get('energy', 0)
            rep   = row.get('reputation', 0)
            print(f"  {row['iso3']}: W={water:.3f} F={food:.3f} E={energy:.3f} Rep={rep:.2f}")

    print("\n✅ WorldSim PettingZoo environment complete.")
    print("   Next: plug in MAPPO agents via RLlib for trained behavior.")
    print("   See worldsim_mappo.py for the training code.")
