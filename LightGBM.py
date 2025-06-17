# random_forest_to_lightgbm.py

from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier  # Importamos LightGBM

# ---------------------------------------------------
# 1. CONEXIÓN A MONGODB ATLAS (CON TIMEOUTS)
# ---------------------------------------------------

# URI de conexión (usuario y contraseña incluidos)
# 🚨 IMPORTANTE: Mantén esta URI oculta y no la subas a repositorios públicos.
uri = "mongodb+srv://benja15mz:RCwovLqffkTViFfl@database.5iimvyd.mongodb.net/"

# Creamos el cliente indicando timeouts para evitar bloqueos largos
client = MongoClient(
    uri,
    serverSelectionTimeoutMS=5000,  # 5 segundos para encontrar un servidor primario
    socketTimeoutMS=20000           # 20 segundos de timeout para operaciones de lectura/escritura
)

# ------------------------------------------
# 2. DEFINIR BASES Y COLECCIONES EN MONGO
# ------------------------------------------

# Base de datos “GPS” y su colección “Gps”
db_gps = client["GPS"]
collection_gps = db_gps["Gps"]

# Base de datos “Turnos” y su colección “Turnos”
db_turnos = client["Turnos"]
collection_turnos = db_turnos["Turnos"]

# ---------------------------------------------------
# 3. EXTRAER DATOS (PROYECCIÓN), LECTURA A PANDAS
# ---------------------------------------------------

# Proyección para la colección GPS: solo los campos necesarios
gps_projection = {
    "_id": 0,
    "vehiculo_id": 1,
    "timestamp": 1,
    "velocidad": 1,
    "temperatura_motor": 1,
    "nivel_combustible": 1,
    "estado_motor": 1,
    "alerta.tipo": 1
}

try:
    # Obtener total de documentos GPS
    total_gps_docs = collection_gps.count_documents({})
    # Tomar una muestra del 1% si hay datos
    sample_size = int(total_gps_docs * 0.01) if total_gps_docs > 0 else 0

    if sample_size > 0:
        # Pipeline de agregación: $sample y luego proyección
        pipeline_sample = [
            {"$sample": {"size": sample_size}},
            {"$project": gps_projection}
        ]
        cursor_gps = collection_gps.aggregate(pipeline_sample)
    else:
        # Si no hay datos, o no queremos muestrear, traemos todo con proyección
        cursor_gps = collection_gps.find(projection=gps_projection)

    df_gps = pd.DataFrame(list(cursor_gps))
    print(f"Datos GPS cargados: {len(df_gps)} registros (muestra 1%)")
except Exception as e:
    print("Error al extraer datos GPS:", e)
    df_gps = pd.DataFrame()

# Proyección para la colección Turnos: solo los campos necesarios
turnos_projection = {
    "_id": 0,
    "vehiculo_id": 1,
    "operador_id": 1,
    "fecha": 1,
    "hora_inicio": 1,
    "hora_fin": 1
}

try:
    cursor_turnos = collection_turnos.find(projection=turnos_projection)
    df_turnos = pd.DataFrame(list(cursor_turnos))
    print(f"Datos Turnos cargados: {len(df_turnos)} registros")
except Exception as e:
    print("Error al extraer datos Turnos:", e)
    df_turnos = pd.DataFrame()

# ---------------------------------------------------
# 4. LIMPieza y FORMATO DE DATOS
# ---------------------------------------------------

if not df_gps.empty:
    # Convertir 'timestamp' a datetime
    df_gps["timestamp"] = pd.to_datetime(df_gps["timestamp"], errors="coerce")
    # Extraer 'alert_type' desde 'alerta.tipo' (subdocumento)
    df_gps["alert_type"] = df_gps["alerta"].apply(
        lambda a: a.get("tipo") if isinstance(a, dict) else None
    )
    # Codificar 'estado_motor' a numérico: encendido=1, apagado=0
    df_gps["estado_motor_encendido"] = df_gps["estado_motor"].map({
        "encendido": 1,
        "apagado": 0
    }).fillna(0).astype(int)
else:
    print("No hay datos GPS para procesar.")

if not df_turnos.empty:
    # Convertir 'fecha' a datetime.date
    df_turnos["fecha"] = pd.to_datetime(df_turnos["fecha"], errors="coerce").dt.date
    # Crear las columnas 'turno_inicio' y 'turno_fin' como datetimes completos
    df_turnos["turno_inicio"] = pd.to_datetime(
        df_turnos["fecha"].astype(str) + " " + df_turnos["hora_inicio"],
        format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    df_turnos["turno_fin"] = pd.to_datetime(
        df_turnos["fecha"].astype(str) + " " + df_turnos["hora_fin"],
        format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
else:
    print("No hay datos Turnos para procesar.")

# ---------------------------------------------------
# 5. CREACIÓN DE FEATURES DERIVADAS
# ---------------------------------------------------

if not df_gps.empty and not df_turnos.empty:
    # Merge por 'vehiculo_id' para saber si cada registro GPS está dentro de su turno
    df_merged = pd.merge(
        df_gps,
        df_turnos[["vehiculo_id", "operador_id", "turno_inicio", "turno_fin"]],
        how="left",
        on="vehiculo_id"
    )
    # 'en_turno' = 1 si 'timestamp' cae entre 'turno_inicio' y 'turno_fin'
    df_merged["en_turno"] = (
        (df_merged["timestamp"] >= df_merged["turno_inicio"]) &
        (df_merged["timestamp"] <= df_merged["turno_fin"])
    ).astype(int)
    # 'fuera_de_turno' = 1 si no está en turno
    df_merged["fuera_de_turno"] = (df_merged["en_turno"] == 0).astype(int)
    # Umbrales para velocidad, temperatura y combustible
    df_merged["velocidad_alta"] = (df_merged["velocidad"] > 80).astype(int)
    df_merged["temperatura_alta"] = (df_merged["temperatura_motor"] > 85).astype(int)
    df_merged["combustible_bajo"] = (df_merged["nivel_combustible"] < 15).astype(int)
    # Mapear 'alert_type' a variable numérica 'tipo_alerta': ninguna=0, leve=1, grave=2, critica=3
    mapping_alerta = {
        "ninguna": 0,
        "leve": 1,
        "grave": 2,
        "critica": 3
    }
    df_merged["tipo_alerta"] = df_merged["alert_type"].map(mapping_alerta)
    # Filtrar solo filas con 'tipo_alerta' válido
    df_modelo = df_merged.dropna(subset=["tipo_alerta"]).copy()
    df_modelo["tipo_alerta"] = df_modelo["tipo_alerta"].astype(int)
else:
    df_modelo = pd.DataFrame()
    print("No se pudo crear df_modelo porque faltan datos GPS o Turnos.")

# ---------------------------------------------------
# 6. PREPARAR MATRIZ DE FEATURES (X) Y TARGET (y)
# ---------------------------------------------------

if not df_modelo.empty:
    features = [
        "velocidad",
        "temperatura_motor",
        "nivel_combustible",
        "estado_motor_encendido",
        "fuera_de_turno",
        "velocidad_alta",
        "temperatura_alta",
        "combustible_bajo"
    ]
    X = df_modelo[features]
    y = df_modelo["tipo_alerta"]

    # Dividir en train/test (80%/20%), estratificando según y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
else:
    print("No se pudo crear conjuntos de entrenamiento y prueba porque df_modelo está vacío.")
    X_train = X_test = y_train = y_test = pd.DataFrame()

# ---------------------------------------------------
# 7. VALIDACIÓN CRUZADA ESTRATIFICADA Y GRID SEARCH PARA LightGBM
# ---------------------------------------------------

# Definir el clasificador LightGBM (sin parámetros ajustados todavía)
lgbm = LGBMClassifier(random_state=42)

# Parámetros a explorar en GridSearchCV para LightGBM
param_grid = {
    "n_estimators": [100, 200],        # cantidad de árboles en el conjunto
    "max_depth": [10, 20, -1],         # profundidad máxima; -1 indica sin límite
    "learning_rate": [0.1, 0.05],      # tasa de aprendizaje
    "num_leaves": [31, 64],            # número de hojas máximo en cada árbol
    "class_weight": ["balanced"]       # equilibrar clases automáticamente
}

# Estrategia de validación cruzada: StratifiedKFold con 5 pliegues
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Crear el GridSearchCV con scoring en f1_weighted para balancear importancia de cada clase
grid_search = GridSearchCV(
    estimator=lgbm,
    param_grid=param_grid,
    scoring="f1_weighted",    # métrica que pondera el F1 por soporte de cada clase
    cv=cv,
    n_jobs=-1,                # usar todos los núcleos disponibles
    verbose=2                 # mostrar progreso de búsqueda
)

# Ejecutar la búsqueda de hiperparámetros solo si tenemos datos
if not X_train.empty:
    print("\nIniciando GridSearchCV para LightGBM...")
    grid_search.fit(X_train, y_train)

    # Mostrar los mejores parámetros encontrados
    print("\n--- Mejores Parámetros Encontrados en LightGBM ---")
    print(grid_search.best_params_)

    # Obtener el mejor modelo ajustado
    best_lgbm = grid_search.best_estimator_
else:
    best_lgbm = None
    print("No se entrenó GridSearchCV porque no hay datos de entrenamiento.")

# ---------------------------------------------------
# 8. EVALUAR MODELO ÓPTIMO EN CONJUNTO DE PRUEBA
# ---------------------------------------------------

if best_lgbm is not None and not X_test.empty:
    print("\nEvaluando el mejor modelo LightGBM en el conjunto de prueba...")
    y_pred = best_lgbm.predict(X_test)

    labels = [0, 1, 2, 3]
    target_names = ["ninguna", "leve", "grave", "critica"]

    print("\n=== Reporte de Clasificación LightGBM (Test Set) ===")
    print(classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0  # Evita errores si alguna clase no aparece
    ))
else:
    print("No se pudo evaluar el modelo porque falta best_lgbm o X_test está vacío.")
