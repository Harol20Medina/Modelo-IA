# =======================================
# Versión 12 - Modelo IA con RandomForest
# Predicción de tipo de alerta vehicular
# =======================================

# -----------------------------
# 1. IMPORTACIÓN DE LIBRERÍAS
# -----------------------------
from pymongo import MongoClient         # Para conectarse a la base de datos MongoDB
import pandas as pd                     # Para manipulación de datos en estructuras tipo DataFrame
import numpy as np                      # Para operaciones numéricas
from datetime import datetime           # Para manejar fechas y horas
from sklearn.ensemble import RandomForestClassifier  # Clasificador basado en árboles
from sklearn.model_selection import train_test_split, GridSearchCV  # División y ajuste de hiperparámetros
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score  # Métricas
from imblearn.over_sampling import SMOTE  # Para balancear las clases minoritarias

# -----------------------------
# 2. CONEXIÓN A MONGODB
# -----------------------------
uri = "mongodb+srv://benja15mz:123@database.5iimvyd.mongodb.net/"
client = MongoClient(uri)
db = client["GPS"]
col = db["Gps"]

# -----------------------------
# 3. CARGA DE DATOS DESDE MONGODB
# -----------------------------
# Se seleccionan las columnas necesarias con proyección
projection = {
    "_id": 0,
    "vehiculo_id": 1,
    "timestamp": 1,
    "velocidad": 1,
    "temperatura_motor": 1,
    "nivel_combustible": 1,
    "estado_motor": 1,
    "alerta": 1
}

# Se convierte el cursor en un DataFrame
df = pd.DataFrame(list(col.find({}, projection)))
print(f"✅ Datos cargados: {len(df)} registros")

# -----------------------------
# 4. PROCESAMIENTO DE DATOS Y FEATURE ENGINEERING
# -----------------------------
# Conversión de timestamp a datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# Extracción de hora y día de la semana
df["hora"] = df["timestamp"].dt.hour
df["dia_semana"] = df["timestamp"].dt.weekday

# Extracción del tipo de alerta
df["alert_type"] = df["alerta"].apply(lambda x: x.get("tipo") if isinstance(x, dict) else None)

# Conversión del estado del motor a binario
df["estado_motor_bin"] = df["estado_motor"].map({"encendido": 1, "apagado": 0}).fillna(0).astype(int)

# Creación de nuevas variables booleanas
df["velocidad_alta"] = (df["velocidad"] > 80).astype(int)
df["temperatura_alta"] = (df["temperatura_motor"] > 85).astype(int)
df["combustible_bajo"] = (df["nivel_combustible"] < 15).astype(int)

# Variable cruzada: velocidad x temperatura
df["velocidad_x_temp"] = df["velocidad"] * df["temperatura_motor"]

# Mapeo del tipo de alerta a valores numéricos
map_alert = {"leve": 1, "grave": 2, "critica": 3}
df["tipo_alerta"] = df["alert_type"].map(map_alert)

# -----------------------------
# 5. FILTRADO DE REGISTROS INCONSISTENTES O RUIDOSOS
# -----------------------------
# Se conservan solo registros con clases válidas
df = df[df["tipo_alerta"].isin([1, 2, 3])]

# Eliminación de registros inconsistentes con alertas graves o críticas en condiciones normales
df = df[
    ~((df["velocidad"] < 30) &
      (df["temperatura_motor"] < 75) &
      (df["nivel_combustible"] > 40) &
      (df["tipo_alerta"] >= 2))
]

# -----------------------------
# 6. SELECCIÓN DE CARACTERÍSTICAS Y VARIABLES TARGET
# -----------------------------
features = [
    "velocidad", "temperatura_motor", "nivel_combustible", "estado_motor_bin",
    "velocidad_alta", "temperatura_alta", "combustible_bajo",
    "hora", "dia_semana", "velocidad_x_temp"
]

# Eliminación de registros con valores nulos
df = df.dropna(subset=features + ["tipo_alerta"])

# Separación en variables independientes (X) y variable dependiente (y)
X = df[features]
y = df["tipo_alerta"]

# -----------------------------
# 7. BALANCEO DE CLASES CON SMOTE
# -----------------------------
# Se aplica sobremuestreo a las clases minoritarias para evitar sesgo en el modelo
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X, y)

# -----------------------------
# 8. DIVISIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_sm, y_sm, test_size=0.2, random_state=42, stratify=y_sm
)

# -----------------------------
# 9. ENTRENAMIENTO DEL MODELO CON GRIDSEARCH
# -----------------------------
# Se define la grilla de hiperparámetros para búsqueda
params = {
    "n_estimators": [100],       # Número de árboles
    "max_depth": [10, 20],       # Profundidad máxima
    "min_samples_split": [2],    # Mínimo de muestras para dividir un nodo
    "min_samples_leaf": [1]      # Mínimo de muestras por hoja
}

# Se entrena un RandomForest con pesos balanceados
rf = RandomForestClassifier(random_state=42, class_weight="balanced")
grid = GridSearchCV(rf, params, cv=3, scoring="f1_macro", verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Se selecciona el mejor modelo
best_model = grid.best_estimator_

# -----------------------------
# 10. EVALUACIÓN DEL MODELO
# -----------------------------
y_pred = best_model.predict(X_test)               # Predicciones
y_proba = best_model.predict_proba(X_test)        # Probabilidades por clase

print("\n✅ Mejores hiperparámetros:", grid.best_params_)
print("\n🔎 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Reporte de clasificación:")
print(classification_report(y_test, y_pred, zero_division=0))
print("🧱 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# Cálculo del AUC-ROC para problemas multiclase
try:
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    print(f"📊 AUC-ROC: {auc:.4f}")
except:
    print("⚠️ No se pudo calcular AUC.")
