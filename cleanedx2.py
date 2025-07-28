import pandas as pd

# Ruta original del archivo
file_path = r"C:\Users\Usuario\OneDrive - FRUTTO FOODS\scriptspython\price-analysis\data\market_cleaned.xlsx"

# 1. Cargar datos
df = pd.read_excel(file_path)

# 2. Renombrar columnas
df.rename(columns={
    'Date': 'cotization_date',
    'OG/CV': 'Organic'
}, inplace=True)

# 3. Eliminar columnas no deseadas
drop_cols = ['Volume?', 'volume_num', 'ProductClean', 'Size']
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# 4. Eliminar filas con valores faltantes
df.dropna(how='any', inplace=True)

# 5. Filtrar errores en Location: quitar dígitos
df = df[~df['Location'].astype(str).str.contains(r'\d', na=False)]

# 6. Estandarizar Location con mapping personalizado
location_map = {
    'arvin':'Arvin','bakersfield':'Bakersfield',
    'ca':'California','ca.':'California','california':'California',
    'canada':'Canada','coachella':'Coachella','dublin':'Dublin','delaware':'Delaware',
    'deliver la':'Los Angeles','delivered':'Los Angeles','el centro ca':'El Centro',
    'florida':'Florida',
    'forth worth':'Fort Worth','fortworth':'Fort Worth','ftw':'Fort Worth','fort worth':'Fort Worth',
    'fresno':'Fresno','fresno ca':'Fresno',
    'georgia':'Georgia','gainesville':'Gainesville',
    'hendersonville nc':'Hendersonville',
    'la':'Los Angeles','los angeles':'Los Angeles','los angeles delivered':'Los Angeles',
    'laredo':'Laredo','leamington':'Leamington','linden':'Linden',
    'marfa':'Marfa','marfa texas':'Marfa','mendota':'Mendota',
    'miami':'Miami','michigan':'Michigan',
    'nogales':'Nogales',
    'new jersey':'New Jersey',
    'north carolina':'North Carolina','north caroline':'North Carolina',
    'oceano ca':'Oceano','ontario':'Ontario','ontario leamington':'Ontario','ontario, ca.':'Ontario',
    'others':'Others','other':'Others',
    'oxnard ca':'Oxnard','pa':'Pennsylvania','patterson ca':'Patterson','penn terminal':'Penn Terminal',
    'pompano':'Pompano','rose hill nc':'Rose Hill','san diego':'San Diego','sandiego':'San Diego',
    'santa maria':'Santa Maria','santa maria ca':'Santa Maria','santa maría':'Santa Maria',
    'santa maría ca':'Santa Maria','santa maria cal':'Santa Maria','sta ma calif':'Santa Maria',
    'salinas':'Salinas','san juan bautista':'San Juan Bautista','selma / coachella':'Selma','somis, ca.':'Somis',
    'stockton ca':'Stockton','swedesboro nj':'Swedesboro','tampa':'Tampa','thermal':'Thermal',
    'vernon':'Vernon','watsonville':'Watsonville','wigham':'Wigham','wigham, ga':'Wigham','wigham, ga.':'Wigham',
    'midway':'Midway','canada':'Canada','california':'California','texas':'Texas','texas.':'Texas'
}
# Normalizar y mapear sin perder nombres existentes
df['Location'] = (
    df['Location'].astype(str)
      .str.strip()
      .str.lower()
      .map(location_map)
      .fillna(df['Location'])
)

# 7. Guardar archivo limpio
output_path = r"C:\Users\Usuario\OneDrive - FRUTTO FOODS\scriptspython\price-analysis\data\market_cleaned_cleaned.xlsx"

df.to_excel(output_path, index=False)
print("✅ Archivo limpiado y Location estandarizado; guardado en:", output_path)
