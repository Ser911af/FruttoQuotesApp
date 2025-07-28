import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar los datos
def cargar_datos():
    cotizaciones = pd.read_csv('data/Market - Consolidado(Consolidado).csv')
    ventas = pd.read_csv('data/reportSalesFrutto_2025-07-24.csv')
    return cotizaciones, ventas

# Limpiar los datos
def limpiar_datos(cotizaciones, ventas):
    cotizaciones = cotizaciones.drop(columns=['Price', 'Date', 'Volume?'], errors='ignore')
    ventas = ventas.drop(columns=['Price', 'Date', 'Volume?'], errors='ignore')
    return cotizaciones, ventas

# Entrenar el modelo de regresión
def entrenar_modelo(cotizaciones):
    # Convertir columnas categóricas a numéricas
    cotizaciones = pd.get_dummies(cotizaciones, columns=['producto', 'tamaño', 'ciudad', 'OG/CV', 'vendor'], drop_first=True)
    
    # Definir características y variable objetivo
    X = cotizaciones.drop('precio', axis=1)
    y = cotizaciones['precio']
    
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Predecir y evaluar el modelo
    y_pred = modelo.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f'R²: {r2:.2f}, MAE: {mae:.2f}')
    
    return modelo

if __name__ == "__main__":
    cotizaciones, ventas = cargar_datos()
    cotizaciones, ventas = limpiar_datos(cotizaciones, ventas)
    modelo = entrenar_modelo(cotizaciones)