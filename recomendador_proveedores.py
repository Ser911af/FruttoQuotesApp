import pandas as pd

# Cargar los archivos CSV
def cargar_datos():
    cotizaciones = pd.read_csv('data/Market - Consolidado(Consolidado).csv')
    ventas = pd.read_csv('data/reportSalesFrutto_2025-07-24.csv')
    return cotizaciones, ventas

# Limpiar los datos
def limpiar_datos(cotizaciones, ventas):
    cotizaciones = cotizaciones.drop(columns=['Price', 'Date', 'Volume?'], errors='ignore')
    ventas = ventas.drop(columns=['Price', 'Date', 'Volume?'], errors='ignore')
    return cotizaciones, ventas

# Función para buscar proveedores
def buscar_vendors(producto, ciudad):
    cotizaciones, ventas = cargar_datos()
    cotizaciones, ventas = limpiar_datos(cotizaciones, ventas)

    # Filtrar cotizaciones por producto y ciudad
    resultados = cotizaciones[(cotizaciones['Producto'] == producto) & (cotizaciones['Ciudad'] == ciudad)]

    if resultados.empty:
        return "No se encontraron proveedores para este producto en la ciudad especificada."

    # Calcular estadísticas
    ultimo_precio = resultados['Precio'].iloc[-1]
    promedio_precio = resultados['Precio'].mean()
    fechas = resultados['Fecha'].tolist()
    og_cv = resultados['OG/CV'].tolist()
    tamanio = resultados['Tamaño'].tolist()
    volumen = resultados['Volumen'].tolist()

    # Crear un DataFrame para mostrar resultados
    df_resultados = pd.DataFrame({
        'Proveedor': resultados['Proveedor'],
        'Último Precio': ultimo_precio,
        'Precio Promedio': promedio_precio,
        'Fechas': [fechas],
        'OG/CV': [og_cv],
        'Tamaño': [tamanio],
        'Volumen': [volumen]
    })

    # Ordenar resultados por precio más bajo
    df_resultados = df_resultados.sort_values(by='Último Precio')

    return df_resultados

# Ejemplo de uso
if __name__ == "__main__":
    producto = 'EjemploProducto'
    ciudad = 'EjemploCiudad'
    resultados = buscar_vendors(producto, ciudad)
    print(resultados)