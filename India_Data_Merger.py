"""
WorldSim – India State-Level Dataset Merger
============================================
Merges 6 heterogeneous datasets into a single analysis-ready DataFrame
seeded with real resource data for the WorldSim simulation engine.

Datasets expected (update FILE_PATHS below to point to your actual files):
  1. crop_production.csv       – State, District, Crop_Year, Season, Crop, Area, Production
  2. energy_capacity.csv       – Country, State, Date, Category, Subcategory, Variable, Unit, Value, …
  3. rainfall_subdivision.csv  – Subdivision, Year, JAN…DEC, ANNUAL, seasonal cols
  4. rainfall_district.csv     – STATE_UT_NAME, DISTRICT, JAN…DEC, ANNUAL, seasonal cols
  5. population.csv            – state, 1951, 1961, …, 2021, data
  6. per_capita_income.csv     – state, 1990, 1991, …, 2021, CATEGORY

Output: worldsim_merged.csv  (one row per state per year, wide format)
"""

import re
import warnings
import numpy as np
import pandas as pd
from difflib import get_close_matches

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  FILE PATHS  ← update these to your actual file locations
# ─────────────────────────────────────────────────────────────────────────────
FILE_PATHS = {
    "crop":        "crop_production.csv",
    "energy":      "energy_capacity.csv",
    "rain_sub":    "rainfall_subdivision.csv",
    "rain_dist":   "rainfall_district.csv",
    "population":  "population.csv",
    "pci":         "per_capita_income.csv",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CANONICAL STATE NAME MAP
#     All variations across datasets → one clean name
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_STATES = {
    # Andaman & Nicobar
    "andaman and nicobar islands":      "Andaman & Nicobar Islands",
    "andaman & nicobar islands":        "Andaman & Nicobar Islands",
    "andaman and nicobar island":       "Andaman & Nicobar Islands",
    "andaman & nicobar island":         "Andaman & Nicobar Islands",
    "andaman and nicobar":              "Andaman & Nicobar Islands",
    "a & n islands":                    "Andaman & Nicobar Islands",

    # Andhra Pradesh
    "andhra pradesh":                   "Andhra Pradesh",

    # Arunachal Pradesh
    "arunachal pradesh":                "Arunachal Pradesh",

    # Assam
    "assam":                            "Assam",

    # Bihar
    "bihar":                            "Bihar",

    # Chandigarh
    "chandigarh":                       "Chandigarh",

    # Chhattisgarh
    "chhattisgarh":                     "Chhattisgarh",
    "chattisgarh":                      "Chhattisgarh",

    # Dadra and Nagar Haveli
    "dadra and nagar haveli":           "Dadra & Nagar Haveli",
    "dadra & nagar haveli":             "Dadra & Nagar Haveli",
    "dadra and nagar haveli and daman and diu": "Dadra & Nagar Haveli",

    # Daman and Diu
    "daman and diu":                    "Daman & Diu",
    "daman & diu":                      "Daman & Diu",

    # Delhi
    "delhi":                            "Delhi",
    "nct of delhi":                     "Delhi",
    "national capital territory of delhi": "Delhi",

    # Goa
    "goa":                              "Goa",

    # Gujarat
    "gujarat":                          "Gujarat",

    # Haryana
    "haryana":                          "Haryana",

    # Himachal Pradesh
    "himachal pradesh":                 "Himachal Pradesh",

    # Jammu & Kashmir
    "jammu and kashmir":                "Jammu & Kashmir",
    "jammu & kashmir":                  "Jammu & Kashmir",
    "j&k":                              "Jammu & Kashmir",

    # Jharkhand
    "jharkhand":                        "Jharkhand",

    # Karnataka
    "karnataka":                        "Karnataka",

    # Kerala
    "kerala":                           "Kerala",

    # Lakshadweep
    "lakshadweep":                      "Lakshadweep",

    # Madhya Pradesh
    "madhya pradesh":                   "Madhya Pradesh",

    # Maharashtra
    "maharashtra":                      "Maharashtra",

    # Manipur
    "manipur":                          "Manipur",

    # Meghalaya
    "meghalaya":                        "Meghalaya",

    # Mizoram
    "mizoram":                          "Mizoram",

    # Nagaland
    "nagaland":                         "Nagaland",

    # Odisha
    "odisha":                           "Odisha",
    "orissa":                           "Odisha",

    # Puducherry
    "puducherry":                       "Puducherry",
    "pondicherry":                      "Puducherry",

    # Punjab
    "punjab":                           "Punjab",

    # Rajasthan
    "rajasthan":                        "Rajasthan",

    # Sikkim
    "sikkim":                           "Sikkim",

    # Tamil Nadu
    "tamil nadu":                       "Tamil Nadu",

    # Telangana
    "telangana":                        "Telangana",

    # Tripura
    "tripura":                          "Tripura",

    # Uttar Pradesh
    "uttar pradesh":                    "Uttar Pradesh",

    # Uttarakhand
    "uttarakhand":                      "Uttarakhand",
    "uttaranchal":                      "Uttarakhand",

    # West Bengal
    "west bengal":                      "West Bengal",
}


def normalize_state(name: str) -> str:
    """
    Lowercase + strip punctuation, then look up in CANONICAL_STATES.
    Falls back to fuzzy matching if no exact key is found.
    Returns the canonical name, or the cleaned original if no match.
    """
    if pd.isna(name):
        return np.nan
    cleaned = re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned in CANONICAL_STATES:
        return CANONICAL_STATES[cleaned]

    # Fuzzy fallback
    matches = get_close_matches(cleaned, CANONICAL_STATES.keys(), n=1, cutoff=0.80)
    if matches:
        return CANONICAL_STATES[matches[0]]

    # Return title-cased original so it's at least readable
    return str(name).strip().title()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET LOADERS
#     Each returns a clean DataFrame with a "state" and "year" column.
# ─────────────────────────────────────────────────────────────────────────────

def load_crop(path: str) -> pd.DataFrame:
    """
    Aggregate crop production to state × year level.
    Returns: state, year, total_crop_area_ha, total_crop_production_tonnes,
             n_crop_types
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["state"] = df["State_Name"].apply(normalize_state)
    df["year"]  = pd.to_numeric(df["Crop_Year"], errors="coerce").astype("Int64")

    agg = (
        df.groupby(["state", "year"])
        .agg(
            total_crop_area_ha        = ("Area",       "sum"),
            total_crop_production_t   = ("Production", "sum"),
            n_crop_types              = ("Crop",       "nunique"),
        )
        .reset_index()
    )
    return agg


def load_energy(path: str) -> pd.DataFrame:
    """
    Pivot energy capacity to state × year.
    Keeps MW capacity columns per fuel type (Clean, Fossil, Renewable, Solar, Wind, …).
    Returns wide-format with prefix 'energy_'.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df["state"] = df["State"].apply(normalize_state)
    df["year"]  = pd.to_datetime(df["Date"], errors="coerce").dt.year.astype("Int64")

    # Filter to Capacity rows only, Variable == MW
    cap = df[
        (df["Category"].str.lower() == "capacity") &
        (df["Unit"].str.upper() == "MW")
    ].copy()

    # Use Subcategory as the column pivot key
    cap["col"] = "energy_" + cap["Subcategory"].str.strip().str.replace(r"\W+", "_", regex=True).str.lower()

    pivot = (
        cap.groupby(["state", "year", "col"])["Value"]
        .sum()
        .unstack("col")
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def load_rainfall_subdivision(path: str) -> pd.DataFrame:
    """
    State-level annual + seasonal rainfall from subdivision data.
    Aggregates subdivisions → state by mean.
    Returns: state, year, rain_annual_mm, rain_junjul_mm, rain_octdec_mm, …
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()

    # Try to detect the state/subdivision column
    state_col = next((c for c in df.columns if "SUBDIV" in c or "STATE" in c), None)
    if state_col is None:
        raise ValueError("Cannot find subdivision/state column in rainfall_subdivision file.")

    df["state"] = df[state_col].apply(normalize_state)
    df["year"]  = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Rename aggregated columns with prefix
    agg = (
        df.groupby(["state", "year"])[numeric_cols]
        .mean()
        .reset_index()
    )

    rename_map = {}
    for c in numeric_cols:
        if c == "YEAR":
            continue
        rename_map[c] = "rain_sub_" + c.lower()
    agg.rename(columns=rename_map, inplace=True)

    # Clean up duplicate year col if present
    if "rain_sub_year" in agg.columns:
        agg.drop(columns=["rain_sub_year"], inplace=True)

    return agg


def load_rainfall_district(path: str) -> pd.DataFrame:
    """
    Aggregate district-level rainfall to state level (mean across districts).
    Returns: state, rain_dist_jan, …, rain_dist_annual, rain_dist_junjul, …
    (No year column — this is a long-run average per district, not time-series.)
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.upper()

    state_col = next(
        (c for c in df.columns if "STATE" in c or "UT" in c), df.columns[0]
    )
    df["state"] = df[state_col].apply(normalize_state)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    agg = (
        df.groupby("state")[numeric_cols]
        .mean()
        .reset_index()
    )
    rename_map = {c: "rain_dist_" + c.lower() for c in numeric_cols}
    agg.rename(columns=rename_map, inplace=True)
    return agg


def load_population(path: str) -> pd.DataFrame:
    """
    Melt wide census data (one column per decade) to long format.
    Returns: state, year, population_thousands
    Interpolates missing decades using cubic spline for continuity.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Find the state column
    state_col = next(
        (c for c in df.columns if "state" in c.lower()), df.columns[0]
    )
    df["state"] = df[state_col].apply(normalize_state)

    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    id_cols   = ["state"] + [c for c in df.columns if c not in year_cols and c != state_col]

    melted = df.melt(
        id_vars=["state"],
        value_vars=year_cols,
        var_name="year",
        value_name="population_thousands",
    )
    melted["year"] = melted["year"].astype(int)
    melted["population_thousands"] = pd.to_numeric(
        melted["population_thousands"], errors="coerce"
    )
    return melted[["state", "year", "population_thousands"]]


def load_pci(path: str) -> pd.DataFrame:
    """
    Melt per-capita income wide format to long.
    Returns: state, year, per_capita_income_inr
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    state_col = next(
        (c for c in df.columns if "state" in c.lower()), df.columns[0]
    )
    df["state"] = df[state_col].apply(normalize_state)

    year_cols = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]

    melted = df.melt(
        id_vars=["state"],
        value_vars=year_cols,
        var_name="year",
        value_name="per_capita_income_inr",
    )
    melted["year"] = melted["year"].astype(int)
    melted["per_capita_income_inr"] = pd.to_numeric(
        melted["per_capita_income_inr"], errors="coerce"
    )
    return melted[["state", "year", "per_capita_income_inr"]]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MERGE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_base_index(
    year_start: int = 2000,
    year_end:   int = 2023,
) -> pd.DataFrame:
    """
    Create a complete (state × year) grid as the merge backbone.
    Every dataset is left-joined onto this so no state/year combination is lost.
    """
    states = list(set(CANONICAL_STATES.values()))
    years  = list(range(year_start, year_end + 1))
    base   = pd.MultiIndex.from_product([states, years], names=["state", "year"])
    return pd.DataFrame(index=base).reset_index()


def interpolate_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Census data is only available every ~10 years.
    Interpolate to annual resolution per state using cubic spline.
    """
    result = []
    for state, grp in df.groupby("state"):
        grp = grp.set_index("year").sort_index()
        # Reindex to full annual range
        full_years = range(grp.index.min(), grp.index.max() + 1)
        grp = grp.reindex(full_years)
        grp["population_thousands"] = (
            grp["population_thousands"]
            .interpolate(method="cubicspline")
            .clip(lower=0)
        )
        grp["state"] = state
        grp = grp.reset_index().rename(columns={"index": "year"})
        result.append(grp)
    return pd.concat(result, ignore_index=True)


def merge_all(
    year_start: int = 2000,
    year_end:   int = 2023,
    files: dict = FILE_PATHS,
) -> pd.DataFrame:
    """
    Master merge function.
    1. Load each dataset
    2. Normalize state names
    3. Join everything onto a (state × year) backbone
    4. Forward-fill/interpolate where appropriate
    5. Return tidy merged DataFrame
    """
    print("📦 Loading datasets …")

    # --- Load ---
    crop   = load_crop(files["crop"])
    energy = load_energy(files["energy"])
    r_sub  = load_rainfall_subdivision(files["rain_sub"])
    r_dist = load_rainfall_district(files["rain_dist"])
    pop    = interpolate_population(load_population(files["population"]))
    pci    = load_pci(files["pci"])

    print(f"   Crop:       {len(crop):,} rows  | {crop['state'].nunique()} states")
    print(f"   Energy:     {len(energy):,} rows  | {energy['state'].nunique()} states")
    print(f"   Rain(sub):  {len(r_sub):,} rows  | {r_sub['state'].nunique()} states")
    print(f"   Rain(dist): {len(r_dist):,} rows  | {r_dist['state'].nunique()} states")
    print(f"   Population: {len(pop):,} rows  | {pop['state'].nunique()} states")
    print(f"   PCI:        {len(pci):,} rows  | {pci['state'].nunique()} states")

    # --- Backbone ---
    print("\n🔗 Building (state × year) backbone …")
    base = build_base_index(year_start, year_end)

    # --- Left joins on state + year ---
    # Datasets that are state×year
    for label, df in [("crop", crop), ("energy", energy),
                      ("rain_sub", r_sub), ("pop", pop), ("pci", pci)]:
        base = base.merge(df, on=["state", "year"], how="left")
        print(f"   Merged {label}: {base.shape}")

    # Rainfall district has no year — merge on state only
    base = base.merge(r_dist, on="state", how="left")
    print(f"   Merged rain_dist: {base.shape}")

    # --- Gap-fill with forward-fill within state ---
    time_varying_cols = [c for c in base.columns if c not in ("state", "year")]
    base = base.sort_values(["state", "year"])
    base[time_varying_cols] = (
        base.groupby("state")[time_varying_cols]
        .transform(lambda s: s.ffill().bfill())
    )

    print(f"\n✅ Final merged shape: {base.shape}")
    print(f"   States: {base['state'].nunique()}")
    print(f"   Year range: {base['year'].min()} – {base['year'].max()}")
    print(f"   Columns: {len(base.columns)}")

    # --- Basic missingness report ---
    miss = base.isnull().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    if not miss.empty:
        print("\n⚠️  Columns with remaining nulls (% missing):")
        for col, pct in miss.head(10).items():
            print(f"   {col:<45} {pct:.1%}")
        if len(miss) > 10:
            print(f"   … and {len(miss)-10} more")

    return base


# ─────────────────────────────────────────────────────────────────────────────
# 4.  WORLDSIM SNAPSHOT  – latest-year resource profile per state
#     This is the format the simulation engine seeds each agent with.
# ─────────────────────────────────────────────────────────────────────────────

def build_worldsim_snapshot(merged: pd.DataFrame, snapshot_year: int = 2021) -> pd.DataFrame:
    """
    Extract a single-year cross-section for the WorldSim initialization.
    Normalizes key resource columns to [0, 1] stress indices.
    """
    snap = merged[merged["year"] == snapshot_year].copy()

    # ── Resource stress indices (higher = more stressed) ──────────────────
    # Water stress: inverse of rainfall (lower rain → higher stress)
    if "rain_sub_annual" in snap.columns:
        rain_max = snap["rain_sub_annual"].max()
        snap["water_stress_index"] = 1 - snap["rain_sub_annual"].fillna(0) / (rain_max + 1e-9)

    # Food stress: inverse of production per capita
    if "total_crop_production_t" in snap.columns and "population_thousands" in snap.columns:
        snap["food_prod_per_capita"] = (
            snap["total_crop_production_t"] /
            (snap["population_thousands"].replace(0, np.nan) * 1000)
        )
        fmax = snap["food_prod_per_capita"].max()
        snap["food_stress_index"] = 1 - snap["food_prod_per_capita"].fillna(0) / (fmax + 1e-9)

    # Energy abundance: total installed capacity per capita (MW / million people)
    energy_total_cols = [c for c in snap.columns if "energy_aggregate_fuel" in c and "clean" in c.lower()]
    if energy_total_cols:
        snap["energy_capacity_mw_percapita"] = (
            snap[energy_total_cols].sum(axis=1) /
            (snap["population_thousands"].replace(0, np.nan) / 1000)
        )

    # Wealth index (PCI normalized)
    if "per_capita_income_inr" in snap.columns:
        pci_max = snap["per_capita_income_inr"].max()
        snap["wealth_index"] = snap["per_capita_income_inr"].fillna(0) / (pci_max + 1e-9)

    return snap.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Full time-series merge
    merged = merge_all(year_start=2000, year_end=2023)
    merged.to_csv("worldsim_merged.csv", index=False)
    print("\n💾 Saved → worldsim_merged.csv")

    # WorldSim seed snapshot (most recent year)
    snapshot = build_worldsim_snapshot(merged, snapshot_year=2021)
    snapshot.to_csv("worldsim_snapshot_2021.csv", index=False)
    print("💾 Saved → worldsim_snapshot_2021.csv")

    # Quick preview of seed data
    key_cols = [
        "state",
        "water_stress_index",
        "food_stress_index",
        "wealth_index",
        "population_thousands",
        "total_crop_production_t",
    ]
    available = [c for c in key_cols if c in snapshot.columns]
    print("\n📊 WorldSim Seed Preview (2021):")
    print(snapshot[available].sort_values("water_stress_index", ascending=False).to_string(index=False))
