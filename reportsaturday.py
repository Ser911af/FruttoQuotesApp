# == Dependencias ==
# Asegúrate de instalar los siguientes paquetes en tu entorno virtual:
#    pip install pandas openpyxl xlsxwriter

import os
import pandas as pd

# == Configuración de rutas ==
input_file = r"C:\Users\Usuario\Downloads\sales-by-item-report-Sat_-26-Jul-2025-13_31_57-GMT.xlsx"
output_dir = r"C:\Users\Usuario\OneDrive - FRUTTO FOODS\scriptspython\price-analysis\datereport"

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# == Lectura de datos ==
try:
    df = pd.read_excel(input_file)
except Exception as e:
    raise RuntimeError(f"Error al leer el archivo de entrada: {e}")

# Columnas disponibles
columns = df.columns.tolist()
if 'Sales Rep' not in columns:
    raise KeyError(f"No se encontró la columna 'Sales Rep'. Columnas disponibles: {columns}")
if 'Product' not in columns:
    raise KeyError(f"No se encontró la columna 'Product'. Columnas disponibles: {columns}")

# Función para sanear nombres para archivos

def sanitize(name: str) -> str:
    safe = ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in str(name))
    return safe.strip().replace(' ', '_')

# == Mapeo robusto de productos a commodities ==
# Define la lista de commodities usados en buyer_products:
buyer_products = {
    'Angie': ['Tomato', 'Cucumber', 'Soft squash'],
    'Gabriela': ['Eggplant', 'Melons', 'Avocado', 'Hard Squash', 'Lemon', 'Pineapple'],
    'John': [
        'Green Beans', 'Bell Pepper', 'Berries', 'Broccoli', 'Hot Peppers',
        'Celery', 'Minisweets', 'Cauliflower', 'Corn'
    ]
}
commodities = [c.lower() for prods in buyer_products.values() for c in prods]

# Función para extraer commodity de cada valor granular de Product
def map_commodity(prod: str) -> str:
    p = prod.lower().strip()
    # Tomato
    if p.startswith('tomato'):
        return 'Tomato'
    # Eggplant
    if p.startswith('eggplant'):
        return 'Eggplant'
    # Cucumber
    if 'cucumber' in p:
        return 'Cucumber'
    # Bell Pepper
    if p.startswith('bell pepper'):
        return 'Bell Pepper'
    # Minisweets (Pepper Mini Sweet)
    if p.startswith('pepper mini sweet'):
        return 'Minisweets'
    # Squash Hard vs Soft
    if 'squash' in p:
        if '- hard' in p:
            return 'Hard Squash'
        if '- fancy' in p:
            return 'Soft squash'
    # Otros commodities directos
    for comm in commodities:
        if p.startswith(comm):
            return comm.title() if comm.islower() else comm
    # Por defecto, Other
    return 'Other'

# Crear columna Commodity
df['Commodity'] = df['Product'].astype(str).apply(map_commodity)

# Mostrar mapeo faltante para ajuste de buyer_products
productos_existentes = set(df['Commodity'].unique())
print(f"Commodities detectados: {productos_existentes}")
for buyer, prods in buyer_products.items():
    faltantes = set(prods) - productos_existentes
    if faltantes:
        print(f"[⚠️] Para {buyer}, estos commodities no fueron detectados: {faltantes}")

# == Exportar por Sales Rep con formato tabla ==
for rep, group in df.groupby('Sales Rep'):
    safe_rep = sanitize(rep)
    file_path = os.path.join(output_dir, f"sales_by_item_{safe_rep}.xlsx")

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        group.to_excel(writer, index=False, sheet_name='Data')
        worksheet = writer.sheets['Data']
        max_row, max_col = group.shape
        worksheet.add_table(0, 0, max_row, max_col - 1, {
            'columns': [{'header': col} for col in group.columns]
        })
    print(f"Reporte para Sales Rep '{rep}' guardado en: {file_path}")

# == Exportar por Compras (Buyers) con formato tabla ==
for buyer, products in buyer_products.items():
    df_buyer = df[df['Commodity'].isin(products)]
    safe_buyer = sanitize(buyer)
    file_path = os.path.join(output_dir, f"sales_by_buyer_{safe_buyer}.xlsx")

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        df_buyer.to_excel(writer, index=False, sheet_name='Data')
        worksheet = writer.sheets['Data']
        max_row, max_col = df_buyer.shape
        worksheet.add_table(0, 0, max_row, max_col - 1, {
            'columns': [{'header': col} for col in df_buyer.columns]
        })
    print(f"Reporte para Buyer '{buyer}' guardado en: {file_path}")

print("Proceso completado: reportes generados por Sales Rep y Buyers.")
