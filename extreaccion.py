from pymongo import MongoClient
import pandas as pd

# ⚠️ Conexión a MongoDB Atlas
# Asegúrate de mantener esta URI segura si contiene usuario y contraseña reales.
# No la compartas públicamente ni la subas a repositorios sin protección.
uri = "mongodb+srv://benja15mz:RCwovLqffkTViFfl@database.5iimvyd.mongodb.net/"

# Crear el cliente de conexión a MongoDB
client = MongoClient(uri)

# -----------------------------------
# 0. Verificar bases de datos y colecciones
# -----------------------------------

print("=== Listado de bases de datos en el clúster ===")
dbs = client.list_database_names()
print(dbs)  # Lista todas las bases de datos disponibles

# Según tu mensaje, la base se llama "GPS" (todo mayúsculas) y la colección "Gps"
db_name_gps = "GPS"
collection_name_gps = "Gps"

if db_name_gps in dbs:
    print(f"\n=== Listado de colecciones en la base '{db_name_gps}' ===")
    cols_gps = client[db_name_gps].list_collection_names()
    print(cols_gps)
else:
    print(f"\nLa base '{db_name_gps}' no aparece entre las bases de datos.")

# La otra base es "Turnos" con colección también llamada "Turnos"
db_name_turnos = "Turnos"
collection_name_turnos = "Turnos"

if db_name_turnos in dbs:
    print(f"\n=== Listado de colecciones en la base '{db_name_turnos}' ===")
    cols_turnos = client[db_name_turnos].list_collection_names()
    print(cols_turnos)
else:
    print(f"\nLa base '{db_name_turnos}' no aparece entre las bases de datos.")

# ------------------------------------------
# 1. Extracción de datos desde la base "GPS"
# ------------------------------------------

# Acceder a la base "GPS"
db_gps = client[db_name_gps]

# Verificar que la colección "Gps" exista en esa base
if collection_name_gps in db_gps.list_collection_names():
    collection_gps = db_gps[collection_name_gps]
else:
    print(f"\nLa colección '{collection_name_gps}' no existe en la base '{db_name_gps}'.")
    collection_gps = None

if collection_gps is not None:
    # Contar documentos en "GPS.Gps"
    count_gps = collection_gps.count_documents({})
    print(f"\nCantidad de documentos en '{db_name_gps}.{collection_name_gps}': {count_gps}")

    # Mostrar hasta 5 documentos de ejemplo
    print("Ejemplo de documentos en 'Gps' (máx 5):")
    for doc in collection_gps.find().limit(5):
        print(doc)

    # Convertir a DataFrame si hay datos
    if count_gps > 0:
        df_gps = pd.DataFrame(list(collection_gps.find()))
    else:
        df_gps = pd.DataFrame()

    # Mostrar primeros registros o indicar vacío
    if not df_gps.empty:
        print("\nPrimeros registros de df_gps:")
        print(df_gps.head())
    else:
        print("\nATENCIÓN: df_gps está vacío. No se cargaron documentos de 'GPS.Gps'.")
else:
    df_gps = pd.DataFrame()

# ----------------------------------------------
# 2. Extracción de datos desde la base "Turnos"
# ----------------------------------------------

# Acceder a la base "Turnos"
db_turnos = client[db_name_turnos]

# Verificar que la colección "Turnos" exista en esa base
if collection_name_turnos in db_turnos.list_collection_names():
    collection_turnos = db_turnos[collection_name_turnos]
else:
    print(f"\nLa colección '{collection_name_turnos}' no existe en la base '{db_name_turnos}'.")
    collection_turnos = None

if collection_turnos is not None:
    # Contar documentos en "Turnos.Turnos"
    count_turnos = collection_turnos.count_documents({})
    print(f"\nCantidad de documentos en '{db_name_turnos}.{collection_name_turnos}': {count_turnos}")

    # Mostrar hasta 5 documentos de ejemplo
    print("Ejemplo de documentos en 'Turnos' (máx 5):")
    for doc in collection_turnos.find().limit(5):
        print(doc)

    # Convertir a DataFrame si hay datos
    if count_turnos > 0:
        df_turnos = pd.DataFrame(list(collection_turnos.find()))
    else:
        df_turnos = pd.DataFrame()

    # Mostrar primeros registros o indicar vacío
    if not df_turnos.empty:
        print("\nPrimeros registros de df_turnos:")
        print(df_turnos.head())
    else:
        print("\nATENCIÓN: df_turnos está vacío. No se cargaron documentos de 'Turnos.Turnos'.")
else:
    df_turnos = pd.DataFrame()
