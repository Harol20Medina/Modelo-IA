from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# ---------------------------------------------------
# 1. CONEXIÓN A MONGODB ATLAS (CON TIMEOUTS)
# ---------------------------------------------------
uri = "mongodb+srv://benja15mz:RCwovLqffkTViFfl@database.5iimvyd.mongodb.net/"

client = MongoClient(
    uri,
    serverSelectionTimeoutMS=5000,  # Timeout 5 segundos para conexión al servidor
    socketTimeoutMS=20000           # Timeout 20 segundos para lectura/escritura
)

# ---------------------------------------------------
# 2. DEFINIR BASES Y COLECCIONES EN MONGO
# ---------------------------------------------------
db_gps = client["GPS"]
collection_gps = db_gps["Gps"]

db_turnos = client["Turnos"]
collection_turnos = db_turnos["Turnos"]

# ---------------------------------------------------
# 3. EXTRAER DATOS (PROYECCIÓN), LECTURA A PANDAS
# ---------------------------------------------------
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
    total_gps_docs = collection_gps.count_documents({})
    sample_size = int(total_gps_docs * 0.01) if total_gps_docs > 0 else 0

    if sample_size > 0:
        pipeline_sample = [
            {"$sample": {"size": sample_size}},
            {"$project": gps_projection}
        ]
        cursor_gps = collection_gps.aggregate(pipeline_sample)
    else:
        cursor_gps = collection_gps.find(projection=gps_projection)

    df_gps = pd.DataFrame(list(cursor_gps))
    print(f"Datos GPS cargados: {len(df_gps)} registros (muestra 1%)")
except Exception as e:
    print("Error al extraer datos GPS:", e)
    df_gps = pd.DataFrame()

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
# 4. LIMPIEZA Y FORMATO DE DATOS
# ---------------------------------------------------
if not df_gps.empty:
    df_gps["timestamp"] = pd.to_datetime(df_gps["timestamp"], errors="coerce")

    # Extraer tipo de alerta de subdocumento
    df_gps["alert_type"] = df_gps["alerta"].apply(
        lambda a: a.get("tipo") if isinstance(a, dict) else None
    )

    # Mapear estado motor a valores numéricos: encendido=1, apagado=0
    df_gps["estado_motor_encendido"] = df_gps["estado_motor"].map({
        "encendido": 1,
        "apagado": 0
    }).fillna(0).astype(int)
else:
    print("No hay datos GPS para procesar.")

if not df_turnos.empty:
    df_turnos["fecha"] = pd.to_datetime(df_turnos["fecha"], errors="coerce").dt.date

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
    df_merged = pd.merge(
        df_gps,
        df_turnos[["vehiculo_id", "operador_id", "turno_inicio", "turno_fin"]],
        how="left",
        on="vehiculo_id"
    )

    # Variable binaria: si el registro GPS está dentro del turno
    df_merged["en_turno"] = (
        (df_merged["timestamp"] >= df_merged["turno_inicio"]) &
        (df_merged["timestamp"] <= df_merged["turno_fin"])
    ).astype(int)

    df_merged["fuera_de_turno"] = (df_merged["en_turno"] == 0).astype(int)

    # Variables binarias de umbrales de sensores
    df_merged["velocidad_alta"] = (df_merged["velocidad"] > 80).astype(int)
    df_merged["temperatura_alta"] = (df_merged["temperatura_motor"] > 85).astype(int)
    df_merged["combustible_bajo"] = (df_merged["nivel_combustible"] < 15).astype(int)

    # Mapear tipos de alerta a números para target
    mapping_alerta = {
        "ninguna": 0,
        "leve": 1,
        "grave": 2,
        "critica": 3
    }
    df_merged["tipo_alerta"] = df_merged["alert_type"].map(mapping_alerta)

    # Eliminar registros con tipo_alerta nulo para el modelo
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

    # Dividir datos en entrenamiento y prueba (80/20)
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
# 7. VALIDACIÓN CRUZADA ESTRATIFICADA Y GRID SEARCH
# ---------------------------------------------------
base_rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

if not X_train.empty:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=skf,
        scoring="accuracy",  # Métrica para elegir mejor modelo
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_

    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)
else:
    print("No se pudo realizar GridSearchCV porque no hay datos de entrenamiento.")
    best_rf = None

# ---------------------------------------------------
# 8. EVALUACIÓN DEL MODELO (CÁLCULO DE MÉTRICAS)
# ---------------------------------------------------
if best_rf is not None and not X_test.empty:
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)

    # --- MÉTRICA: Accuracy (Precisión global) ---
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}\n")

    # --- MÉTRICAS: Precision, Recall y F1-score ---
    print("Reporte de clasificación (Test) [Precision, Recall, F1-score]:")
    print(classification_report(y_test, y_pred))

    # --- MÉTRICA: Matriz de confusión ---
    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusión:")
    print(cm)

    # --- MÉTRICA: AUC-ROC multiclasificación (One-vs-Rest) ---
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        print(f"AUC-ROC (multi-clase): {auc:.4f}")
    except Exception as e:
        print(f"No se pudo calcular AUC-ROC: {e}")
else:
    print("No se pudo evaluar el modelo por falta de datos o modelo entrenado.")
