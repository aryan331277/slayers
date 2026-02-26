"""
WorldSim — Master Dataset Merge Script
=======================================
Run this on Kaggle. Outputs: worldsim_final.csv

Sources merged:
  1. final_integrated_dataset.csv      — World Bank + OWID backbone
  2. data_cts_violent_and_sexual_crime.xlsx  — UNODC violent crime rates
  3. data_cts_corruption_and_economic_crime.xlsx — UNODC corruption rates
  4. global_water_consumption_2000_2025.csv  — Water consumption + scarcity
  5. EM-DAT xlsx                        — Climate disaster shock counts

Scope: 2000–2018, all countries (not just the 10 WorldSim regions).
The 10 target regions are flagged in a column for easy filtering.
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# PATHS  — adjust if your Kaggle dataset name differs
# ─────────────────────────────────────────────
BASE = "/kaggle/input/datasets/aryankarmore/worldbank/"

PATH_MAIN     = BASE + "final_integrated_dataset.csv"
PATH_VIOLENT  = BASE + "data_cts_violent_and_sexual_crime.xlsx"
PATH_CORRUPT  = BASE + "data_cts_corruption_and_economic_crime (1).xlsx"
PATH_WATER    = BASE + "global_water_consumption_2000_2025.csv"
PATH_EMDAT    = BASE + "public_emdat_custom_request_2026-02-26_fe9fb1e5-dcb8-47f3-8d76-faa724205bac.xlsx"

OUTPUT_PATH   = "/kaggle/working/worldsim_final.csv"

YEAR_MIN = 2000
YEAR_MAX = 2018

# 10 WorldSim target regions (iso3)
WORLDSIM_TARGETS = {
    "EGY": "Egypt",
    "ETH": "Ethiopia",
    "IND": "India",
    "CHN": "China",
    "BRA": "Brazil",
    "DEU": "Germany",
    "USA": "United States",
    "SAU": "Saudi Arabia",
    "NGA": "Nigeria",
    "AUS": "Australia",
}

print("=" * 60)
print("WorldSim Dataset Merge")
print("=" * 60)


# ─────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN MAIN BACKBONE
# ─────────────────────────────────────────────
print("\n[1/6] Loading main backbone...")

main = pd.read_csv(PATH_MAIN)

# Filter year range
main = main[(main["Year"] >= YEAR_MIN) & (main["Year"] <= YEAR_MAX)].copy()

# ── Handle duplicates ──
# The main CSV has 7,256 country-year duplicates from the World Bank / OWID merge.
# Strategy: for each (iso3, Year), average all numeric columns.
# This is safer than dropping arbitrarily — it preserves data from both sources.
print(f"  Rows before dedup: {len(main):,}")

numeric_cols = main.select_dtypes(include=[np.number]).columns.tolist()
# Keep non-numeric metadata from first occurrence
meta_cols = ["country", "iso3", "Year"]

# Average numeric, keep first for strings
main_num = main.groupby(["iso3", "Year"])[numeric_cols].mean()
main_meta = main.groupby(["iso3", "Year"])[["country"]].first()
main = main_meta.join(main_num).reset_index()

print(f"  Rows after dedup:  {len(main):,}")
print(f"  Countries:         {main['iso3'].nunique()}")

# Rename WB columns to human-readable for final output
rename_map = {
    "BN.KLT.DINV.CD":    "fdi_net_inflows_usd",
    "DT.DOD.DECT.CD":    "external_debt_usd",
    "DT.TDS.DECT.EX.ZS": "debt_service_pct_exports",
    "EG.ELC.ACCS.ZS":    "electricity_access_pct",
    "EG.USE.PCAP.KG.OE": "energy_use_per_capita_kgoe",
    "EN.ATM.CO2E.KT":    "co2_emissions_kt",
    "EN.ATM.CO2E.PP.GD": "co2_per_gdp_wb",
    "NE.TRD.GNFS.ZS":    "trade_pct_gdp",
    "NV.AGR.TOTL.ZS":    "agriculture_pct_gdp",
    "NV.IND.TOTL.ZS":    "industry_pct_gdp",
    "NV.SRV.TOTL.ZS":    "services_pct_gdp",
    "NY.GDP.PCAP.CD":    "gdp_per_capita_usd",
    "SE.XPD.TOTL.GD.ZS": "education_spend_pct_gdp",
    "SH.XPD.CHEX.GD.ZS": "health_spend_pct_gdp",
    "SL.EMP.TOTL.SP.ZS": "employment_ratio_pct",
    "SL.EMP.VULN.ZS":    "vulnerable_employment_pct",
    "SL.TLF.TOTL.IN":    "total_labor_force",
    "TM.TAX.MRCH.SM.AR.ZS": "tariff_rate_pct",
}
main = main.rename(columns=rename_map)
print("  ✓ Main backbone ready")


# ─────────────────────────────────────────────
# STEP 2 — VIOLENT CRIME (UNODC)
# ─────────────────────────────────────────────
print("\n[2/6] Processing violent crime data...")

vc_raw = pd.read_excel(PATH_VIOLENT, header=1)
vc_raw.columns = vc_raw.iloc[0]         # real header is in row 0 of data
vc_raw = vc_raw.drop(vc_raw.index[0]).reset_index(drop=True)

# Cast types
vc_raw["Year"]  = pd.to_numeric(vc_raw["Year"],  errors="coerce")
vc_raw["VALUE"] = pd.to_numeric(vc_raw["VALUE"], errors="coerce")

# Filter year range
vc_raw = vc_raw[(vc_raw["Year"] >= YEAR_MIN) & (vc_raw["Year"] <= YEAR_MAX)]

# ── Extract: Homicide rate per 100k (most comparable internationally) ──
# Indicator = "Violent offences", Category = "Serious assault" is available for more
# countries but homicide is the gold standard for international comparison.
# We take BOTH:
#   1. homicide_rate_per_100k  — from Victims indicators, rate unit, Total sex/age
#   2. violent_offence_rate_per_100k — Violent offences, rate, Total sex/age

def extract_crime_rate(df, indicator_contains, category, col_name):
    mask = (
        df["Indicator"].str.contains(indicator_contains, case=False, na=False) &
        df["Category"].str.contains(category, case=False, na=False) &
        (df["Unit of measurement"] == "Rate per 100,000 population") &
        (df["Sex"] == "Total") &
        (df["Age"] == "Total")
    )
    sub = df[mask][["Iso3_code", "Year", "VALUE"]].copy()
    # If multiple rows per country-year (e.g. multiple sources), take mean
    sub = sub.groupby(["Iso3_code", "Year"])["VALUE"].mean().reset_index()
    sub = sub.rename(columns={"Iso3_code": "iso3", "VALUE": col_name})
    return sub

# Violent offences — serious assault rate (broadest coverage)
violent_rate = extract_crime_rate(
    vc_raw,
    indicator_contains="Violent offences",
    category="Serious assault",
    col_name="serious_assault_rate_per_100k"
)

print(f"  Violent offences rate: {len(violent_rate):,} country-year rows, "
      f"{violent_rate['iso3'].nunique()} countries")
print("  ✓ Violent crime ready")


# ─────────────────────────────────────────────
# STEP 3 — CORRUPTION (UNODC)
# ─────────────────────────────────────────────
print("\n[3/6] Processing corruption data...")

cc_raw = pd.read_excel(PATH_CORRUPT, header=0)

cc_raw["Year"]  = pd.to_numeric(cc_raw["Year"],  errors="coerce")
cc_raw["VALUE"] = pd.to_numeric(cc_raw["VALUE"], errors="coerce")
cc_raw = cc_raw[(cc_raw["Year"] >= YEAR_MIN) & (cc_raw["Year"] <= YEAR_MAX)]

# Extract: bribery prevalence rate (best proxy for institutional quality)
# Fallback: corruption offence rate per 100k
bribery = cc_raw[
    (cc_raw["Indicator"] == "Prevalence rate of bribery (%)") &
    (cc_raw["Sex"] == "Total") &
    (cc_raw["Age"] == "Total")
][["Iso3_code","Year","VALUE"]].copy()
bribery = bribery.groupby(["Iso3_code","Year"])["VALUE"].mean().reset_index()
bribery = bribery.rename(columns={"Iso3_code":"iso3", "VALUE":"bribery_prevalence_pct"})

# Corruption offence rate per 100k
corrupt_rate = cc_raw[
    (cc_raw["Indicator"] == "Offences") &
    (cc_raw["Category"].str.contains("Corruption$", na=False)) &  # only top-level Corruption
    (cc_raw["Unit of measurement"] == "Rate per 100,000 population") &
    (cc_raw["Sex"] == "Total") &
    (cc_raw["Age"] == "Total")
][["Iso3_code","Year","VALUE"]].copy()
corrupt_rate = corrupt_rate.groupby(["Iso3_code","Year"])["VALUE"].mean().reset_index()
corrupt_rate = corrupt_rate.rename(columns={"Iso3_code":"iso3", "VALUE":"corruption_offence_rate_per_100k"})

print(f"  Bribery prevalence: {len(bribery):,} rows, {bribery['iso3'].nunique()} countries")
print(f"  Corruption rate:    {len(corrupt_rate):,} rows, {corrupt_rate['iso3'].nunique()} countries")
print("  ✓ Corruption ready")


# ─────────────────────────────────────────────
# STEP 4 — WATER CONSUMPTION
# ─────────────────────────────────────────────
print("\n[4/6] Processing water consumption data...")

water = pd.read_csv(PATH_WATER)
water = water[(water["Year"] >= YEAR_MIN) & (water["Year"] <= YEAR_MAX)].copy()

# ── Problem: no iso3 in this file, only country names ──
# Build a manual name→iso3 mapping for all 150 countries in this file.
# We map to iso3 so the join is clean and unambiguous.

# First: build mapping from the main dataset (country name → iso3)
name_to_iso = main[["country","iso3"]].drop_duplicates()

# The water file uses different name conventions — patch known mismatches
water_name_fixes = {
    "USA":                        "United States",
    "DR Congo":                   "Dem. Rep. Congo",
    "Côte d'Ivoire":              "Cote d'Ivoire",
    "North Korea":                "Korea, Dem. People's Rep.",
    "South Korea":                "Korea, Rep.",
    "Czech Republic":             "Czechia",
    "State of Palestine":         "West Bank and Gaza",
    "Hong Kong":                  "Hong Kong SAR, China",
    "Taiwan":                     "Taiwan, China",
    "Laos":                       "Lao PDR",
    "Syria":                      "Syrian Arab Republic",
    "Iran":                       "Iran, Islamic Rep.",
    "Russia":                     "Russian Federation",
    "Bolivia":                    "Bolivia (Plurinational State of)",
    "Venezuela":                  "Venezuela, RB",
    "Egypt":                      "Egypt, Arab Rep.",
    "South Sudan":                "South Sudan",
    "Moldova":                    "Moldova",
    "Bosnia and Herzegovina":     "Bosnia and Herzegovina",
}

# Create a reverse lookup: water_name → iso3
# Step 1: direct match on country name from main
main_name_iso = dict(zip(main["country"], main["iso3"]))

# Step 2: apply name fixes BEFORE lookup
# We'll map water country names to canonical main names first
def get_iso3(water_country_name):
    # Try direct
    if water_country_name in main_name_iso:
        return main_name_iso[water_country_name]
    # Try via fix map (the fix map keys are water names, values are main names)
    if water_country_name in water_name_fixes:
        canonical = water_name_fixes[water_country_name]
        return main_name_iso.get(canonical, None)
    return None

water["iso3"] = water["Country"].apply(get_iso3)

# Report unmapped
unmapped = water[water["iso3"].isna()]["Country"].unique()
if len(unmapped) > 0:
    print(f"  WARNING: {len(unmapped)} water countries could not be mapped to iso3:")
    for u in unmapped:
        print(f"    - {u}")

# Drop unmapped countries (likely territories / micro-states not in main dataset)
water = water[water["iso3"].notna()].copy()

# Clean up and rename
water_cols = {
    "Total Water Consumption (Billion m3)": "water_consumption_billion_m3",
    "Per Capita Water Use (L/Day)":         "water_use_per_capita_l_day",
    "Agricultural Water Use (%)":           "agri_water_use_pct",
    "Industrial Water Use (%)":             "industrial_water_use_pct",
    "Household Water Use (%)":              "household_water_use_pct",
    "Rainfall Impact (mm)":                 "rainfall_mm",
    "Groundwater Depletion Rate (%)":       "groundwater_depletion_rate_pct",
    "Water Scarcity Level":                 "water_scarcity_level",
}
water = water.rename(columns=water_cols)

# Encode water scarcity level as numeric for modelling
scarcity_map = {"Low": 1, "Moderate": 2, "High": 3, "Extreme": 4, "Critical": 5}
water["water_scarcity_score"] = water["water_scarcity_level"].map(scarcity_map)

water_clean = water[["iso3","Year"] + list(water_cols.values()) + ["water_scarcity_score"]].copy()
# Handle any country-year duplicates (shouldn't exist but be safe)
numeric_water = water_clean.select_dtypes(include=[np.number]).columns.tolist()
water_clean = water_clean.groupby(["iso3","Year"]).agg(
    {col: "mean" if col in numeric_water else "first"
     for col in water_clean.columns if col not in ["iso3","Year"]}
).reset_index()

print(f"  Water rows: {len(water_clean):,}, countries: {water_clean['iso3'].nunique()}")
print("  ✓ Water consumption ready")


# ─────────────────────────────────────────────
# STEP 5 — EM-DAT CLIMATE SHOCKS
# ─────────────────────────────────────────────
print("\n[5/6] Processing EM-DAT climate shocks...")

emdat = pd.read_excel(PATH_EMDAT)

# Filter to relevant climate/natural disaster types and year range
climate_types = ["Drought", "Flood", "Extreme temperature", "Storm", "Wildfire"]
emdat = emdat[
    emdat["Disaster Type"].isin(climate_types) &
    (emdat["Start Year"] >= YEAR_MIN) &
    (emdat["Start Year"] <= YEAR_MAX)
].copy()

emdat = emdat.rename(columns={
    "ISO": "iso3",
    "Start Year": "Year",
    "Total Deaths": "emdat_deaths",
    "Total Affected": "emdat_total_affected",
    "Total Damage ('000 US$)": "emdat_damage_000usd",
    "Disaster Type": "disaster_type",
})

# ── Aggregate to country-year level ──
# We want: count of each disaster type + total deaths + total affected + total damage

# Count of each disaster type per country-year (pivot)
shock_counts = (
    emdat.groupby(["iso3","Year","disaster_type"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)
# Rename columns
shock_counts.columns.name = None
type_rename = {
    "Drought":              "shocks_drought_count",
    "Flood":                "shocks_flood_count",
    "Extreme temperature":  "shocks_heatwave_count",
    "Storm":                "shocks_storm_count",
    "Wildfire":             "shocks_wildfire_count",
}
shock_counts = shock_counts.rename(columns=type_rename)

# Aggregate impact columns
shock_impact = (
    emdat.groupby(["iso3","Year"])
    .agg(
        total_shock_events     = ("disaster_type", "count"),
        emdat_total_deaths     = ("emdat_deaths", "sum"),
        emdat_total_affected   = ("emdat_total_affected", "sum"),
        emdat_total_damage_usd = ("emdat_damage_000usd", "sum"),
    )
    .reset_index()
)
# Convert damage from thousands to millions for readability
shock_impact["emdat_total_damage_usd"] = shock_impact["emdat_total_damage_usd"] * 1000

# Merge counts + impact
emdat_agg = shock_counts.merge(shock_impact, on=["iso3","Year"], how="outer")

# Fill missing shock type counts with 0 (means no events that year)
count_cols = list(type_rename.values())
for col in count_cols:
    if col not in emdat_agg.columns:
        emdat_agg[col] = 0
emdat_agg[count_cols] = emdat_agg[count_cols].fillna(0).astype(int)

print(f"  EM-DAT rows: {len(emdat_agg):,}, countries: {emdat_agg['iso3'].nunique()}")
print("  ✓ EM-DAT climate shocks ready")


# ─────────────────────────────────────────────
# STEP 6 — MERGE ALL SOURCES
# ─────────────────────────────────────────────
print("\n[6/6] Merging all sources on (iso3, Year)...")

df = main.copy()
print(f"  Start:              {len(df):,} rows, {df['iso3'].nunique()} countries")

# ── Merge violent crime ──
df = df.merge(violent_rate, on=["iso3","Year"], how="left")
print(f"  After violent crime: {df['serious_assault_rate_per_100k'].notna().sum():,} matched")

# ── Merge corruption (bribery) ──
df = df.merge(bribery, on=["iso3","Year"], how="left")
print(f"  After bribery:       {df['bribery_prevalence_pct'].notna().sum():,} matched")

# ── Merge corruption (offence rate) ──
df = df.merge(corrupt_rate, on=["iso3","Year"], how="left")
print(f"  After corrupt rate:  {df['corruption_offence_rate_per_100k'].notna().sum():,} matched")

# ── Merge water consumption ──
df = df.merge(water_clean, on=["iso3","Year"], how="left")
print(f"  After water:         {df['water_use_per_capita_l_day'].notna().sum():,} matched")

# ── Merge EM-DAT shocks ──
# EM-DAT is sparse — years with no disasters simply won't appear in emdat_agg
# After merge, fill shock counts with 0 (no disasters that year = 0 shocks)
df = df.merge(emdat_agg, on=["iso3","Year"], how="left")
count_cols_all = list(type_rename.values()) + ["total_shock_events"]
df[count_cols_all] = df[count_cols_all].fillna(0)
# Impact columns (deaths, affected) stay NaN when truly no data
print(f"  After EM-DAT:        {df['total_shock_events'].sum():.0f} total shock events merged")


# ─────────────────────────────────────────────
# POST-MERGE: FLAG + DERIVED COLUMNS
# ─────────────────────────────────────────────
print("\n[Post-merge] Adding derived columns...")

# Flag WorldSim 10 target countries
df["is_worldsim_target"] = df["iso3"].isin(WORLDSIM_TARGETS.keys()).astype(int)

# Composite fragility score (0–1 normalized)
# Built from: vulnerable employment, violent crime, GINI, external debt burden, shock frequency
# All normalized to [0,1] individually then averaged.
# NOTE: NaN-safe — only counts non-null components
def safe_norm(series):
    """Min-max normalize a series, NaN-safe."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)

fragility_components = []
for col in ["vulnerable_employment_pct", "serious_assault_rate_per_100k",
            "gini", "groundwater_depletion_rate_pct", "total_shock_events"]:
    if col in df.columns:
        normed = safe_norm(df[col])
        fragility_components.append(normed)

if fragility_components:
    df["fragility_score"] = pd.concat(fragility_components, axis=1).mean(axis=1, skipna=True)
    print(f"  fragility_score computed from {len(fragility_components)} components")

# Economic structure classification (dominant sector)
# Requires all three sector columns to be non-null
sector_df = df[["agriculture_pct_gdp","industry_pct_gdp","services_pct_gdp"]].copy()
has_all = sector_df.notna().all(axis=1)
dominant = sector_df[has_all].idxmax(axis=1).map({
    "agriculture_pct_gdp": "agrarian",
    "industry_pct_gdp":    "industrial",
    "services_pct_gdp":    "service",
})
df.loc[has_all, "dominant_sector"] = dominant

# Year as integer (sanity check)
df["Year"] = df["Year"].astype(int)


# ─────────────────────────────────────────────
# FINAL OUTPUT
# ─────────────────────────────────────────────
print(f"\nFinal dataset shape: {df.shape}")
print(f"Countries: {df['iso3'].nunique()}")
print(f"Year range: {df['Year'].min()} – {df['Year'].max()}")
print(f"WorldSim targets: {df[df['is_worldsim_target']==1]['iso3'].nunique()}")

# Column summary
print("\nColumn list and missingness for WorldSim targets (post-2000):")
target_df = df[df["is_worldsim_target"] == 1]
for col in df.columns:
    if col in ["country","iso3","Year","is_worldsim_target"]:
        continue
    pct = target_df[col].notna().mean() * 100
    flag = "✓" if pct >= 70 else ("~" if pct >= 40 else "✗")
    print(f"  {flag} {col}: {pct:.0f}% complete in 10 targets")

# Save
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Saved to: {OUTPUT_PATH}")
print(f"   File size: ~{len(df) * len(df.columns) * 8 / 1e6:.1f} MB estimated")
