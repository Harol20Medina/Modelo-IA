from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------
# 1. CONFIGURACIÓN DE LA CONEXIÓN A MONGODB ATLAS
# ---------------------------------------------------

# URI de conexión a MongoDB Atlas (usuario y contraseña incluidos)
# IMPORTANTE: Mantén esta URI oculta y no la subas a repositorios públicos.
uri = "mongodb+srv://benja15mz:RCwovLqffkTViFfl@database.5iimvyd.mongodb.net/"

# Creamos el cliente indicando timeouts:
# - serverSelectionTimeoutMS: tiempo (ms) para que el driver encuentre un servidor.
# - socketTimeoutMS: tiempo (ms) máximo para bloqueos de lectura/escritura de socket.
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
# 3. EXTRAER DATOS (SOLO CAMPOS NECESARIOS) CON PROYECCIÓN
# ---------------------------------------------------

# PROYECCIÓN para la colección GPS: solo extraemos campos que usaremos
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
    # Para muestreo: tomamos solo 10% de documentos al azar
    total_gps_docs = collection_gps.count_documents({})
    sample_size = int(total_gps_docs * 0.1) if total_gps_docs > 0 else 0

    if sample_size > 0:
        pipeline_sample = [
            {"$sample": {"size": sample_size}},
            {"$project": gps_projection}
        ]
        cursor_gps = collection_gps.aggregate(pipeline_sample)
    else:
        cursor_gps = collection_gps.find(projection=gps_projection)

    df_gps = pd.DataFrame(list(cursor_gps))
    print("Datos GPS cargados: {} registros (muestra 10%)".format(len(df_gps)))
except Exception as e:
    print("Error al extraer datos GPS:", e)
    df_gps = pd.DataFrame()

# PROYECCIÓN para la colección Turnos: solo campos para identificar turno
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
    print("Datos Turnos cargados: {} registros".format(len(df_turnos)))
except Exception as e:
    print("Error al extraer datos Turnos:", e)
    df_turnos = pd.DataFrame()

# ---------------------------------------------------
# 4. LIMPieza BÁSICA y PREPARACIÓN DE DATOS
# ---------------------------------------------------

if not df_gps.empty:
    # Convertir 'timestamp' de string a datetime (si viene como string)
    df_gps["timestamp"] = pd.to_datetime(df_gps["timestamp"], errors="coerce")
    # Extraer 'alert_type' desde 'alerta.tipo'
    df_gps["alert_type"] = df_gps["alerta"].apply(
        lambda a: a.get("tipo") if isinstance(a, dict) else None
    )
    # Convertir 'estado_motor' a variable numérica: encendido=1, apagado=0
    df_gps["estado_motor_encendido"] = df_gps["estado_motor"].map({
        "encendido": 1,
        "apagado": 0
    }).fillna(0).astype(int)
else:
    print("No hay datos GPS para procesar.")

if not df_turnos.empty:
    # Convertir 'fecha' a datetime.date
    df_turnos["fecha"] = pd.to_datetime(df_turnos["fecha"], errors="coerce").dt.date
    # Crear 'turno_inicio' y 'turno_fin' como datetime completos
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
    # Unir df_gps con df_turnos por 'vehiculo_id'
    df_merged = pd.merge(
        df_gps,
        df_turnos[["vehiculo_id", "operador_id", "turno_inicio", "turno_fin"]],
        how="left",
        on="vehiculo_id"
    )
    # Determinar si 'timestamp' está dentro de [turno_inicio, turno_fin]
    df_merged["en_turno"] = (
        (df_merged["timestamp"] >= df_merged["turno_inicio"]) &
        (df_merged["timestamp"] <= df_merged["turno_fin"])
    ).astype(int)
    # Crear 'fuera_de_turno' (1 si no está en turno)
    df_merged["fuera_de_turno"] = (df_merged["en_turno"] == 0).astype(int)
    # Umbrales de velocidad, temperatura y combustible
    df_merged["velocidad_alta"] = (df_merged["velocidad"] > 80).astype(int)
    df_merged["temperatura_alta"] = (df_merged["temperatura_motor"] > 85).astype(int)
    df_merged["combustible_bajo"] = (df_merged["nivel_combustible"] < 15).astype(int)
    # Mapear 'alert_type' a numérico 'tipo_alerta'
    mapping_alerta = {
        "ninguna": 0,
        "leve": 1,
        "grave": 2,
        "critica": 3
    }
    df_merged["tipo_alerta"] = df_merged["alert_type"].map(mapping_alerta)
    # Filtrar filas con 'tipo_alerta' válido
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

    # División entrenamiento/prueba 80% / 20%, estratificando según y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------
    # 7. ENTRENAR MODELO Random Forest
    # ---------------------------------------------------

    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_clf.fit(X_train, y_train)

    # ---------------------------------------------------
    # 8. EVALUAR MODELO
    # ---------------------------------------------------

    y_pred = rf_clf.predict(X_test)

    # Aquí indicamos explícitamente las 4 clases (0,1,2,3) y sus nombres
    labels = [0, 1, 2, 3]
    target_names = ["ninguna", "leve", "grave", "critica"]

    print("\nReporte de Clasificación (Test Set):")
    print(classification_report(
        y_test, 
        y_pred, 
        labels=labels, 
        target_names=target_names, 
        zero_division=0  # En caso de que alguna clase no aparezca, no arroje error
    ))
else:
    print("No se entrenó el modelo porque no hay datos en df_modelo.")
