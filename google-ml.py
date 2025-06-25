import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import requests
import os
from datetime import datetime
import warnings
import time
import polyline # Importante: pip install polyline

# --- Configuración Inicial ---
warnings.simplefilter(action='ignore')
pd.set_option('display.max_rows', 100)

# ==============================================================================
# IMPORTANTE: Reemplaza esta cadena con tu propia clave de API
# de la Google Cloud Platform. Debe tener habilitadas:
# Routes API, Roads API, y Distance Matrix API.
# ==============================================================================
API_KEY_GOOGLE = "AQUI_VA_TU_API_KEY"

# Ruta representativa en Camp de Túria (Llíria a Bétera)
ORIGIN_COORDS = "39.6253,-0.5961"
DESTINATION_COORDS = "39.5925,-0.4619"
HISTORICO_CSV_PATH = 'historico_trafico_avanzado.csv'
MODELO_PATH = 'modelo_trafico_avanzado.pkl'

# ==============================================================================
# PARTE A: RECOLECTOR DE DATOS AVANZADO
# ==============================================================================

def obtener_datos_de_ruta_avanzados():
    """
    Orquesta las llamadas a las APIs de Google para obtener datos de ruta,
    límites de velocidad y matriz de distancia.
    """
    print("--- PARTE A: RECOLECTOR DE DATOS AVANZADO ---")
    if API_KEY_GOOGLE == "AQUI_VA_TU_API_KEY":
        print("ADVERTENCIA: No se ha configurado una API Key de Google. No se pueden obtener datos reales.")
        return None

    try:
        # 1. Obtener la ruta principal y la polilínea con Routes API
        print("1/3: Obteniendo ruta y polilínea de Routes API...")
        routes_data = llamar_routes_api(ORIGIN_COORDS, DESTINATION_COORDS)
        if not routes_data or 'routes' not in routes_data or not routes_data['routes']:
            print("Error: Routes API no devolvió una ruta válida.")
            return None
        
        route = routes_data['routes'][0]
        encoded_polyline = route['polyline']['encodedPolyline']
        duracion_con_trafico_seg = int(route['duration'].replace('s', ''))
        distancia_metros = route['distanceMeters']

        # 2. Obtener límites de velocidad con Roads API usando la polilínea
        print("2/3: Obteniendo límites de velocidad de Roads API...")
        velocidad_limite_promedio_kmh = obtener_limite_velocidad_promedio(encoded_polyline)

        # 3. Obtener duración sin tráfico con Distance Matrix API
        print("3/3: Obteniendo datos de Distance Matrix API...")
        matrix_data = llamar_distance_matrix_api(ORIGIN_COORDS, DESTINATION_COORDS)
        if not matrix_data:
            return None
        duracion_sin_trafico_seg = matrix_data['duration']['value']

        # Calcular nuevas características
        factor_congestion = duracion_con_trafico_seg / duracion_sin_trafico_seg if duracion_sin_trafico_seg > 0 else 1.0

        nuevo_registro = {
            'timestamp': datetime.now(),
            'duracion_viaje_seg': duracion_con_trafico_seg,
            'distancia_metros': distancia_metros,
            'velocidad_limite_promedio_kmh': velocidad_limite_promedio_kmh,
            'factor_congestion': round(factor_congestion, 2)
        }
        
        # Guardar en el CSV
        df_nuevo = pd.DataFrame([nuevo_registro])
        df_nuevo.to_csv(HISTORICO_CSV_PATH, mode='a', header=not os.path.exists(HISTORICO_CSV_PATH), index=False)
        print(f"Éxito. Nuevo registro guardado: {nuevo_registro}")
        return nuevo_registro

    except requests.exceptions.RequestException as e:
        print(f"Error de red al llamar a las APIs de Google: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error de formato en la respuesta de la API: {e}")
        return None
    except Exception as e:
        print(f"Un error inesperado ocurrió: {e}")
        return None


def llamar_routes_api(origin, destination):
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': API_KEY_GOOGLE,
        'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline'
    }
    payload = {
        "origin": {"location": {"latLng": {"latitude": float(origin.split(',')[0]), "longitude": float(origin.split(',')[1])}}},
        "destination": {"location": {"latLng": {"latitude": float(destination.split(',')[0]), "longitude": float(destination.split(',')[1])}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()

def obtener_limite_velocidad_promedio(encoded_polyline_str):
    """Decodifica una polilínea y obtiene el límite de velocidad promedio de la Roads API."""
    path = polyline.decode(encoded_polyline_str)
    # Roads API tiene un límite de 100 puntos por llamada
    # Por simplicidad, tomamos hasta 100 puntos espaciados uniformemente
    if len(path) > 100:
        indices = np.linspace(0, len(path) - 1, 100, dtype=int)
        path = [path[i] for i in indices]

    path_str = "|".join([f"{lat},{lon}" for lat, lon in path])
    
    url = f"https://roads.googleapis.com/v1/speedLimits?path={path_str}&key={API_KEY_GOOGLE}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    if 'speedLimits' not in data or not data['speedLimits']:
        return 0.0 # Retornar 0 si no se encontraron límites de velocidad

    limites = [sl['speedLimit'] for sl in data['speedLimits']]
    return round(np.mean(limites), 2) if limites else 0.0


def llamar_distance_matrix_api(origin, destination):
    url = (f"https://maps.googleapis.com/maps/api/distancematrix/json?"
           f"origins={origin}&destinations={destination}&mode=driving&key={API_KEY_GOOGLE}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if data['status'] == 'OK' and data['rows'][0]['elements'][0]['status'] == 'OK':
        return data['rows'][0]['elements'][0]
    else:
        print(f"Error en Distance Matrix API: {data.get('error_message', data['status'])}")
        return None

# ==============================================================================
# PARTE B: ENTRENADOR Y PRONOSTICADOR DE ML
# ==============================================================================

def entrenar_y_predecir():
    """Carga datos históricos enriquecidos, entrena un modelo y predice el tráfico."""
    print("\n--- PARTE B: ENTRENADOR Y PRONOSTICADOR DE ML (AVANZADO) ---")
    
    # --- PASO 1: Cargar datos ---
    if not os.path.exists(HISTORICO_CSV_PATH):
        print(f"Error: El archivo '{HISTORICO_CSV_PATH}' no existe. Ejecute el recolector primero.")
        return
        
    df = pd.read_csv(HISTORICO_CSV_PATH, parse_dates=['timestamp'])
    
    if len(df) < 30:
        print(f"ADVERTENCIA: Hay muy pocos datos ({len(df)} registros). La predicción no será precisa.")

    # --- PASO 2: Ingeniería de Características ---
    df['hora'] = df['timestamp'].dt.hour
    df['dia_semana'] = df['timestamp'].dt.dayofweek
    df['es_finde'] = (df['dia_semana'] >= 5).astype(int)
    
    features = ['hora', 'dia_semana', 'es_finde', 'distancia_metros', 'velocidad_limite_promedio_kmh']
    target = 'duracion_viaje_seg'
    
    X = df[features]
    y = df[target]

    if X.empty or y.empty:
        print("No hay suficientes datos para entrenar el modelo.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- PASO 3: Entrenar el modelo ---
    print("PASO 3: Entrenando el modelo RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1, min_samples_leaf=3, max_features='sqrt')
    model.fit(X_train, y_train)
    joblib.dump(model, MODELO_PATH)
    print(f"Modelo entrenado y guardado como '{MODELO_PATH}'.\n")

    # --- PASO 4: Evaluar el modelo ---
    if not y_test.empty:
        print("PASO 4: Evaluando el rendimiento del modelo...")
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"  - Error Medio Absoluto (MAE): {mae:.2f} segundos (~{int(mae/60)} min)")
        print(f"  - Coeficiente R²: {r2:.2f}\n")
    
    # --- PASO 5: Pronosticar las próximas 24 horas ---
    print("PASO 5: Pronosticando las próximas 24 horas...")
    future_timestamps = pd.date_range(start=datetime.now(), periods=24, freq='h')
    df_futuro = pd.DataFrame({'timestamp': future_timestamps})
    df_futuro['hora'] = df_futuro['timestamp'].dt.hour
    df_futuro['dia_semana'] = df_futuro['timestamp'].dt.dayofweek
    df_futuro['es_finde'] = (df_futuro['dia_semana'] >= 5).astype(int)
    
    # Para el pronóstico, usamos los valores promedio/constantes de las otras características
    df_futuro['distancia_metros'] = df['distancia_metros'].iloc[0] if not df.empty else 0
    df_futuro['velocidad_limite_promedio_kmh'] = df['velocidad_limite_promedio_kmh'].mean() if not df.empty else 0

    X_futuro = df_futuro[features]
    pronostico = model.predict(X_futuro)
    df_futuro['duracion_predicha_seg'] = pronostico.astype(int)
    df_futuro['duracion_predicha_min'] = (df_futuro['duracion_predicha_seg'] / 60).round(1)

    print("\n--- Pronóstico de Duración de Viaje para 'Camp de Túria' (Llíria -> Bétera) ---")
    print(df_futuro[['timestamp', 'duracion_predicha_min']].to_string(index=False))


# --- INICIO DE LA EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    # 1. Ejecutar el recolector para obtener el dato más reciente
    obtener_datos_de_ruta_avanzados()
    
    # 2. Entrenar con todos los datos disponibles y predecir el futuro
    entrenar_y_predecir()
