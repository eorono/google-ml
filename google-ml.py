import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import requests # Para hacer llamadas a la API
import os
from datetime import datetime
import warnings

# --- Configuración Inicial ---
warnings.simplefilter(action='ignore')
pd.set_option('display.max_rows', 50)

# ==============================================================================
# IMPORTANTE: Reemplaza esta cadena con tu propia clave de API
# de la Google Cloud Platform.
# ==============================================================================
API_KEY_GOOGLE = "AQUI_VA_TU_API_KEY"

# Ruta representativa en Camp de Túria (Llíria a Bétera)
ORIGIN_COORDS = {"latitude": 39.6253, "longitude": -0.5961}
DESTINATION_COORDS = {"latitude": 39.5925, "longitude": -0.4619}
HISTORICO_CSV_PATH = 'historico_trafico_google.csv'

# ==============================================================================
# PARTE A: RECOLECTOR DE DATOS DE TRÁFICO EN TIEMPO REAL
# ==============================================================================

def obtener_y_guardar_trafico_actual():
    """
    Llama a la Google Routes API para obtener la duración del viaje actual
    y la guarda en el archivo CSV histórico.
    """
    print("--- PARTE A: RECOLECTOR DE DATOS ---")
    if API_KEY_GOOGLE == "AQUI_VA_TU_API_KEY":
        print("ADVERTENCIA: No se ha configurado una API Key de Google. No se pueden obtener datos reales.")
        return

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': API_KEY_GOOGLE,
        'X-Goog-FieldMask': 'routes.duration,routes.staticDuration'
    }
    payload = {
        "origin": {"location": {"latLng": ORIGIN_COORDS}},
        "destination": {"location": {"latLng": DESTINATION_COORDS}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }

    try:
        print(f"Obteniendo datos de tráfico actuales para la ruta Llíria -> Bétera...")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status() # Lanza un error si la petición falla (ej. 4xx o 5xx)

        data = response.json()
        
        # Extraer la duración del primer resultado de la ruta
        # 'duration' incluye el tráfico, 'staticDuration' es sin tráfico.
        duration_con_trafico_str = data['routes'][0]['duration']
        static_duration_str = data['routes'][0]['staticDuration']
        
        # Convertir de string (ej. "962s") a segundos en número
        duration_con_trafico_seg = int(duration_con_trafico_str.replace('s', ''))
        
        # Crear un nuevo registro
        nuevo_registro = {
            'timestamp': datetime.now(),
            'duracion_viaje_seg': duration_con_trafico_seg
        }
        
        # Guardar en el CSV
        df_nuevo = pd.DataFrame([nuevo_registro])
        # Si el archivo no existe, lo crea con cabecera. Si existe, añade la fila sin cabecera.
        df_nuevo.to_csv(HISTORICO_CSV_PATH, mode='a', header=not os.path.exists(HISTORICO_CSV_PATH), index=False)
        
        print(f"Éxito. Nuevo registro guardado en '{HISTORICO_CSV_PATH}': {nuevo_registro['duracion_viaje_seg']} segundos.")

    except requests.exceptions.RequestException as e:
        print(f"Error al llamar a la API de Google: {e}")
    except KeyError:
        print(f"Error: La respuesta de la API no tuvo el formato esperado. Respuesta: {response.text}")


# ==============================================================================
# PARTE B: ENTRENADOR Y PRONOSTICADOR DE ML
# ==============================================================================

def entrenar_y_predecir():
    """
    Carga los datos históricos, entrena un modelo y predice el tráfico futuro.
    """
    print("\n--- PARTE B: ENTRENADOR Y PRONOSTICADOR DE ML ---")
    
    # --- PASO 1: Cargar los datos históricos ---
    print(f"PASO 1: Cargando datos históricos desde '{HISTORICO_CSV_PATH}'...")
    if not os.path.exists(HISTORICO_CSV_PATH):
        print(f"Error: El archivo '{HISTORICO_CSV_PATH}' no existe. Ejecute el recolector primero para generar datos.")
        # Creamos un archivo de ejemplo para que el script no falle la primera vez
        print("Creando un archivo de ejemplo para demostración...")
        ejemplo_data = {
            'timestamp': pd.to_datetime(['2024-01-01 08:00', '2024-01-01 09:00', '2024-01-01 18:00']),
            'duracion_viaje_seg': [900, 1200, 1100]
        }
        pd.DataFrame(ejemplo_data).to_csv(HISTORICO_CSV_PATH, index=False)
        
    df_historico = pd.read_csv(HISTORICO_CSV_PATH, parse_dates=['timestamp'])
    
    if len(df_historico) < 20:
        print(f"ADVERTENCIA: Hay muy pocos datos ({len(df_historico)} registros). La predicción del modelo no será precisa.")

    # --- PASO 2: Ingeniería de Características ---
    print("PASO 2: Creando características a partir de la fecha/hora...")
    df_historico['hora'] = df_historico['timestamp'].dt.hour
    df_historico['dia_semana'] = df_historico['timestamp'].dt.dayofweek # Lunes=0, Domingo=6
    df_historico['es_finde'] = (df_historico['dia_semana'] >= 5).astype(int)
    
    # Preparar datos para el modelo
    X = df_historico[['hora', 'dia_semana', 'es_finde']]
    y = df_historico['duracion_viaje_seg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- PASO 3: Entrenar el modelo ---
    print("PASO 3: Entrenando el modelo RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2)
    model.fit(X_train, y_train)
    joblib.dump(model, 'modelo_trafico_camp_de_turia.pkl')
    print("Modelo entrenado y guardado como 'modelo_trafico_camp_de_turia.pkl'.\n")

    # --- PASO 4: Evaluar el modelo ---
    print("PASO 4: Evaluando el rendimiento del modelo...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"  - Error Medio Absoluto (MAE): {mae:.2f} segundos. (La predicción se desvía ~{int(mae/60)} min)")
    print(f"  - Coeficiente R²: {r2:.2f}. (Explica el {r2:.0%} de la variabilidad del tráfico)\n")
    
    # --- PASO 5: Pronosticar las próximas 24 horas ---
    print("PASO 5: Usando el modelo para pronosticar las próximas 24 horas...")
    future_timestamps = pd.date_range(start=datetime.now(), periods=24, freq='h')
    df_futuro = pd.DataFrame({'timestamp': future_timestamps})
    df_futuro['hora'] = df_futuro['timestamp'].dt.hour
    df_futuro['dia_semana'] = df_futuro['timestamp'].dt.dayofweek
    df_futuro['es_finde'] = (df_futuro['dia_semana'] >= 5).astype(int)
    
    X_futuro = df_futuro[['hora', 'dia_semana', 'es_finde']]
    pronostico_futuro = model.predict(X_futuro)
    df_futuro['duracion_predicha_seg'] = pronostico_futuro.astype(int)
    df_futuro['duracion_predicha_min'] = (df_futuro['duracion_predicha_seg'] / 60).round(1)

    print("\n--- Pronóstico de Duración de Viaje para 'Camp de Túria' (Llíria -> Bétera) ---")
    print(df_futuro[['timestamp', 'duracion_predicha_min']].to_string(index=False))


# --- INICIO DE LA EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    # 1. Ejecutar el recolector para obtener el dato más reciente (si hay API Key)
    obtener_y_guardar_trafico_actual()
    
    # 2. Entrenar con todos los datos disponibles y predecir el futuro
    entrenar_y_predecir()
