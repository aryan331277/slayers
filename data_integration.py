import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_wdi(path="/kaggle/input/world-development-indicators/wdi-csv-zip-57-mb-/WDIData.csv", 
             indicators=None):
    """
    Load and reshape World Development Indicators data.
    
    Returns long-format DataFrame with country, year, indicator structure.
    """
    print("Loading WDI data...")
    wdi = pd.read_csv(path, low_memory=False)
    
    if indicators is not None:
        wdi = wdi[wdi['Indicator Code'].isin(indicators)]
    
    # Reshape from wide to long format
    year_cols = [col for col in wdi.columns if col.isdigit()]
    id_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    
    wdi = wdi.melt(id_vars=id_cols, value_vars=year_cols,
                   var_name='Year', value_name='Value')
    
    wdi['Year'] = pd.to_numeric(wdi['Year'], errors='coerce')
    wdi = wdi.dropna(subset=['Year'])
    wdi = wdi.rename(columns={'Country Name': 'country', 'Country Code': 'iso3'})
    
    print(f"  Loaded {len(wdi)} observations across {wdi['country'].nunique()} countries")
    return wdi


def load_wiid(path="/kaggle/input/world-income-inequality-database/WIID_06MAY2020.csv"):
    """Load World Income Inequality Database."""
    print("Loading WIID data...")
    wiid = pd.read_csv(path, low_memory=False)
    wiid.columns = wiid.columns.str.lower()
    
    # Identify Gini column
    gini_col = next((c for c in ['gini_reported', 'gini', 'gini_market'] if c in wiid.columns), None)
    if gini_col is None:
        print("  WARNING: No Gini column found")
        return pd.DataFrame()
    
    keep = ['country', 'year', gini_col]
    wiid = wiid[keep].dropna(subset=['country', 'year', gini_col])
    wiid = wiid.rename(columns={gini_col: 'gini', 'year': 'Year'})
    wiid['Year'] = pd.to_numeric(wiid['Year'], errors='coerce')
    wiid = wiid.dropna(subset=['Year'])
    
    print(f"  Loaded {len(wiid)} observations")
    return wiid


def load_owid_co2(path="/kaggle/input/global-co2-and-greenhouse-gas-emissions/owid-co2-data.csv"):
    """Load Our World in Data CO₂ emissions."""
    print("Loading OWID CO₂ data...")
    co2 = pd.read_csv(path, low_memory=False)
    co2 = co2.rename(columns={'country': 'country', 'year': 'Year'})
    
    cols = ['country', 'Year', 'co2', 'co2_per_capita', 'co2_per_gdp', 
            'share_global_co2', 'population', 'gdp']
    available_cols = [c for c in cols if c in co2.columns]
    co2 = co2[available_cols]
    
    print(f"  Loaded {len(co2)} observations")
    return co2


def load_imf_gdp(path="/kaggle/input/imfs-gdp-data-1980-2028-global-trends/gdp.csv"):
    """Load IMF GDP data."""
    print("Loading IMF GDP data...")
    gdp = pd.read_csv(path, low_memory=False)
    gdp.columns = gdp.columns.str.lower()
    
    # Flexible column detection
    year_col = next((c for c in ['year', 'yr', 'time'] if c in gdp.columns), None)
    country_col = next((c for c in ['country', 'country_name', 'nation'] if c in gdp.columns), None)
    value_col = next((c for c in ['gdp', 'value', 'gdp_usd'] if c in gdp.columns), gdp.columns[-1])
    
    if year_col and country_col:
        gdp = gdp.rename(columns={year_col: 'Year', country_col: 'country', value_col: 'gdp_imf'})
        gdp = gdp[['country', 'Year', 'gdp_imf']]
        gdp['Year'] = pd.to_numeric(gdp['Year'], errors='coerce')
        gdp = gdp.dropna(subset=['Year'])
        print(f"  Loaded {len(gdp)} observations")
        return gdp
    else:
        print("  WARNING: Could not identify year/country columns")
        return pd.DataFrame()


def load_trade(path="/kaggle/input/international-trade-database/trade_1988_2021.csv"):
    """Load international trade database."""
    print("Loading trade data...")
    trade = pd.read_csv(path, low_memory=False)
    lc = trade.columns.str.lower()
    
    # Column mapping
    col_map = {}
    mappings = {
        'reporter': ['reporter', 'reporter_name', 'reporter_iso', 'cty1', 'exporter'],
        'partner': ['partner', 'partner_name', 'partner_iso', 'cty2', 'importer'],
        'Year': ['year', 'yr', 'time'],
        'trade_value': ['trade_value', 'value', 'tradevalue', 'valusd', 'trade_value_usd', 'export_value']
    }
    
    for target, candidates in mappings.items():
        for c in candidates:
            if c in lc:
                col_map[trade.columns[list(lc).index(c)]] = target
                break
    
    trade = trade.rename(columns=col_map)
    
    if 'Year' in trade.columns and 'reporter' in trade.columns and 'trade_value' in trade.columns:
        trade['Year'] = pd.to_numeric(trade['Year'], errors='coerce')
        trade = trade.dropna(subset=['Year'])
        result = trade[['reporter', 'Year', 'trade_value']].copy()
        print(f"  Loaded {len(result)} trade observations")
        return result
    else:
        print("  WARNING: Could not identify required columns")
        return pd.DataFrame()


# Indicator codes from World Bank WDI
INDICATORS = [
    'NV.AGR.TOTL.ZS',      # Agriculture value added (% GDP)
    'NV.IND.TOTL.ZS',      # Industry value added (% GDP)
    'NV.SRV.TOTL.ZS',      # Services value added (% GDP)
    'SL.TLF.TOTL.IN',      # Labor force, total
    'SL.EMP.TOTL.SP.ZS',   # Employment to population ratio
    'SL.EMP.VULN.ZS',      # Vulnerable employment (% total)
    'NE.CON.PETC.ZS',      # Household consumption (% GDP)
    'BN.KLT.DINV.CD',      # Foreign direct investment, net inflows
    'DT.DOD.DECT.CD',      # External debt stocks, total
    'DT.TDS.DECT.EX.ZS',   # Debt service (% exports)
    'NE.TRD.GNFS.ZS',      # Trade (% GDP)
    'TM.TAX.MRCH.SM.AR.ZS',# Tariff rate, applied, simple mean
    'EN.ATM.CO2E.KT',      # CO2 emissions (kt)
    'EN.ATM.CO2E.PP.GD',   # CO2 emissions per GDP
    'EG.USE.PCAP.KG.OE',   # Energy use per capita
    'EG.ELC.ACCS.ZS',      # Access to electricity (% population)
    'SH.XPD.CHEX.GD.ZS',   # Health expenditure (% GDP)
    'SE.XPD.TOTL.GD.ZS',   # Education expenditure (% GDP)
    'NY.GDP.PCAP.CD',      # GDP per capita
]

# Load all datasets
print("\n" + "="*80)
print("DATA LOADING PHASE")
print("="*80 + "\n")

wdi = load_wdi(indicators=INDICATORS)
wdi_wide = wdi.pivot_table(index=['country', 'iso3', 'Year'], 
                           columns='Indicator Code', 
                           values='Value').reset_index()

co2 = load_owid_co2()
wiid = load_wiid()
imf = load_imf_gdp()
trade = load_trade()

print("\n" + "="*80)
print("INITIAL DATA SHAPES")
print("="*80)
print(f"WDI Wide: {wdi_wide.shape}")
print(f"CO₂ Data: {co2.shape}")
print(f"WIID: {wiid.shape}")
print(f"IMF GDP: {imf.shape}")
print(f"Trade: {trade.shape}")


print("\n" + "="*80)
print("DATASET INTEGRATION")
print("="*80 + "\n")

df = wdi_wide.copy()
print(f"Base dataset: {df.shape}")

# Merge CO₂ data
if not co2.empty:
    pre_merge = len(df)
    df = df.merge(co2, on=['country', 'Year'], how='left', suffixes=('', '_co2'))
    matched = df['co2'].notna().sum()
    print(f"CO₂ merge: {matched:,} matches ({matched/pre_merge*100:.1f}%)")

# Merge WIID
if not wiid.empty:
    pre_merge = len(df)
    df = df.merge(wiid, on=['country', 'Year'], how='left')
    matched = df['gini'].notna().sum()
    print(f"WIID merge: {matched:,} matches ({matched/pre_merge*100:.1f}%)")

# Merge IMF GDP
if not imf.empty:
    pre_merge = len(df)
    df = df.merge(imf, on=['country', 'Year'], how='left')
    matched = df['gdp_imf'].notna().sum()
    print(f"IMF GDP merge: {matched:,} matches ({matched/pre_merge*100:.1f}%)")

# Aggregate and merge trade data
if not trade.empty:
    print("\nProcessing trade data...")
    trade_agg = trade.groupby(['reporter', 'Year']).agg(
        total_exports=('trade_value', 'sum'),
        trade_partners=('trade_value', 'count')
    ).reset_index()
    trade_agg = trade_agg.rename(columns={'reporter': 'country'})
    
    pre_merge = len(df)
    df = df.merge(trade_agg, on=['country', 'Year'], how='left')
    matched = df['total_exports'].notna().sum()
    print(f"Trade merge: {matched:,} matches ({matched/pre_merge*100:.1f}%)")

print(f"\nFinal integrated dataset: {df.shape}")
print(f"Countries: {df['country'].nunique()}")
print(f"Year range: {df['Year'].min():.0f}-{df['Year'].max():.0f}")


# =============================================================================
# SAVE FINAL INTEGRATED DATASET
# =============================================================================

output_path = "/kaggle/working/final_integrated_dataset.csv"

# Optional: sort for cleanliness
df = df.sort_values(["country", "Year"]).reset_index(drop=True)

# Save to CSV
df.to_csv(output_path, index=False)

print("\n" + "="*80)
print("DATASET EXPORTED SUCCESSFULLY")
print("="*80)
print(f"File saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
