import os
import shutil
import pandas as pd
from difflib import get_close_matches
import re

# 1. Crear carpeta de datos si no existe
DATA_DIR = r"C:\Users\Usuario\OneDrive - FRUTTO FOODS\scriptspython\price-analysis\data"
os.makedirs(DATA_DIR, exist_ok=True)

# 2. Definir rutas origen y destino
SOURCE_MARKET = r"C:\Users\Usuario\Downloads\Market - Consolidado(Consolidado).csv"
SOURCE_SALES  = r"C:\Users\Usuario\OneDrive - FRUTTO FOODS\scriptspython\ssr\data\reportSalesFrutto_2025-07-24.csv"
TARGET_MARKET = os.path.join(DATA_DIR, "market_consolidado.csv")
TARGET_SALES  = os.path.join(DATA_DIR, "sales_report.csv")

# 3. Copiar archivos con verificación
for src, dst in [(SOURCE_MARKET, TARGET_MARKET), (SOURCE_SALES, TARGET_SALES)]:
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"📁 Copiado {os.path.basename(src)} → {dst}")
    else:
        print(f"⚠️ Archivo no encontrado: {src}. Verifica la ruta.")

# 4. Cargar CSVs originales
df_market = pd.read_csv(TARGET_MARKET, encoding='utf-8', skipinitialspace=True)
df_sales  = pd.read_csv(TARGET_SALES,  encoding='utf-8', skipinitialspace=True)

# 5. Renombrar segunda columna en market a 'VendorOrig'
cols = df_market.columns.tolist()
if len(cols) > 1:
    cols[1] = 'VendorOrig'
    df_market.columns = cols

# 6. Unificar columnas base: 'Where' → 'Location', 'Sale Location' y 'Organic' → 'OG/CV'
if 'Where' in df_market.columns:
    df_market.rename(columns={'Where': 'Location'}, inplace=True)
df_sales.rename(columns={'Sale Location': 'Location', 'Organic': 'OG/CV'}, inplace=True)

# 7. Eliminar columnas Unnamed
df_market = df_market.loc[:, ~df_market.columns.str.contains('^Unnamed')]

# 8. Convertir OG/CV a binario
def map_org(x):
    x = str(x).strip().upper()
    if x in ['OG','TRUE','1']: return 1
    if x in ['CV','FALSE','0']: return 0
    return pd.NA

df_market['OG/CV'] = df_market['OG/CV'].map(map_org)
df_sales['OG/CV']  = df_sales['OG/CV'].map(map_org)

# 9. Limpiar y estandarizar volumen
def parse_volume(v):
    s = str(v).strip()
    m = re.match(r"(\d+)\s*([a-zA-Z]+)?", s)
    if not m:
        return pd.Series({'volume_num': pd.NA, 'volume_unit': pd.NA})
    num = int(m.group(1))
    unit_raw = m.group(2).lower() if m.group(2) else ''
    if unit_raw in ['p','palet','palets','pallet','pallets']:
        unit = 'pallet'
    elif unit_raw in ['cs','case','cases','cajas']:
        unit = 'case'
    elif unit_raw in ['load','loads']:
        unit = 'load'
    elif unit_raw in ['bin','bins']:
        unit = 'bin'
    else:
        unit = pd.NA
    return pd.Series({'volume_num': num, 'volume_unit': unit})
volumes = df_market['Volume?'].apply(parse_volume)
df_market = pd.concat([df_market, volumes], axis=1)

# 10. Consolidar 'volume_standard' y eliminar 'Other'
valid_units = ['pallet','case','load','bin']
def standard_volume(row):
    unit = row['volume_unit']
    if pd.isna(unit) or unit not in valid_units:
        return 'Other'
    return unit
df_market['volume_standard'] = df_market.apply(standard_volume, axis=1)
# Eliminar filas con volume_standard == 'Other'
df_market = df_market[df_market['volume_standard'] != 'Other']

# 11. Fuzzy match de VendorOrig → VendorClean y eliminar VendorOrig
sales_vendors = df_sales['Vendor'].dropna().unique().tolist()
def match_vendor(orig):
    o = str(orig).strip().lower()
    matches = get_close_matches(o, [v.lower() for v in sales_vendors], n=1, cutoff=0.8)
    return next((v for v in sales_vendors if v.lower()==matches[0]), orig) if matches else orig
df_market['VendorClean'] = df_market['VendorOrig'].apply(match_vendor)
df_market.drop(columns=['VendorOrig'], inplace=True)

# 12. Fuzzy match de Product → ProductClean (mantener ambas)
sales_products = df_sales['Product'].dropna().unique().tolist()
def match_product(p):
    p0 = str(p).strip().lower()
    matches = get_close_matches(p0, [s.lower() for s in sales_products], n=1, cutoff=0.8)
    return next((s for s in sales_products if s.lower()==matches[0]), p) if matches else p
df_market['ProductClean'] = df_market['Product'].apply(match_product)

# 13. Guardar archivos para Excel
market_csv = os.path.join(DATA_DIR, 'market_cleaned.csv')
market_xlsx = os.path.join(DATA_DIR, 'market_cleaned.xlsx')
sales_csv  = os.path.join(DATA_DIR, 'sales_cleaned.csv')
sales_xlsx = os.path.join(DATA_DIR, 'sales_cleaned.xlsx')
df_market.to_csv(market_csv, index=False)
df_market.to_excel(market_xlsx, index=False)
df_sales.to_csv(sales_csv, index=False)
df_sales.to_excel(sales_xlsx, index=False)
print(f"✅ market_cleaned.csv + XLSX guardados; cols: {df_market.columns.tolist()}")
print(f"✅ sales_cleaned.csv + XLSX guardados; cols: {df_sales.columns.tolist()}")

# 14. Reportes de coincidencias
vendor_matches = df_market[['VendorClean']].drop_duplicates()
product_matches = df_market[['Product','ProductClean']].drop_duplicates()
vendor_matches.to_csv(os.path.join(DATA_DIR,'vendor_matches.csv'), index=False)
product_matches.to_csv(os.path.join(DATA_DIR,'product_matches.csv'), index=False)
print("📊 vendor_matches.csv y product_matches.csv generados")
