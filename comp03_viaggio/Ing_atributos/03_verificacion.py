import duckdb

# Rutas de los datasets
datasets = [
    "gs://sra_electron_bukito3/datasets/competencia_02_FE_v2.parquet",
    "gs://sra_electron_bukito3/datasets/competencia_02_FE_v4.parquet",
    "gs://sra_electron_bukito3/datasets/competencia_03_FE_v2.parquet",
    "gs://sra_electron_bukito3/datasets/competencia_03_FE_v4.parquet"
]

# Conectar a DuckDB
conn = duckdb.connect()

# Configurar acceso a GCS
from google.auth import default
from google.auth.transport.requests import Request

credentials, project = default()
credentials.refresh(Request())
token = credentials.token

conn.execute("INSTALL httpfs;")
conn.execute("LOAD httpfs;")
conn.execute(f"""
    CREATE SECRET (
        TYPE GCS,
        PROVIDER config,
        BEARER_TOKEN '{token}'
    )
""")

print("Cargando datasets para foto_mes = 202106...\n")

# Cargar cada dataset en una tabla temporal
for i, dataset_path in enumerate(datasets, 1):
    dataset_name = f"dataset_{i}"
    
    query = f"""
    CREATE OR REPLACE TABLE {dataset_name} AS
    SELECT numero_de_cliente, target_ternario, target_binario
    FROM read_parquet('{dataset_path}')
    WHERE foto_mes = 202106
    """
    
    conn.execute(query)
    
    count = conn.execute(f"SELECT COUNT(*) FROM {dataset_name}").fetchone()[0]
    print(f"Dataset {i} ({dataset_path.split('/')[-1]}): {count} registros")

print("\n" + "="*80)
print("VERIFICACIÓN DE CONSISTENCIA")
print("="*80 + "\n")

# Verificar que todos los datasets tengan los mismos clientes
print("1. Verificando que todos los datasets tengan los mismos clientes...")

check_query = """
SELECT 
    COUNT(DISTINCT d1.numero_de_cliente) as clientes_d1,
    COUNT(DISTINCT d2.numero_de_cliente) as clientes_d2,
    COUNT(DISTINCT d3.numero_de_cliente) as clientes_d3,
    COUNT(DISTINCT d4.numero_de_cliente) as clientes_d4,
    COUNT(DISTINCT COALESCE(d1.numero_de_cliente, d2.numero_de_cliente, 
                            d3.numero_de_cliente, d4.numero_de_cliente)) as total_unique
FROM dataset_1 d1
FULL OUTER JOIN dataset_2 d2 ON d1.numero_de_cliente = d2.numero_de_cliente
FULL OUTER JOIN dataset_3 d3 ON d1.numero_de_cliente = d3.numero_de_cliente
FULL OUTER JOIN dataset_4 d4 ON d1.numero_de_cliente = d4.numero_de_cliente
"""

result = conn.execute(check_query).fetchone()
print(f"   Dataset 1: {result[0]} clientes")
print(f"   Dataset 2: {result[1]} clientes")
print(f"   Dataset 3: {result[2]} clientes")
print(f"   Dataset 4: {result[3]} clientes")
print(f"   Total único: {result[4]} clientes")

if result[0] == result[1] == result[2] == result[3] == result[4]:
    print("   ✓ Todos los datasets tienen los mismos clientes\n")
else:
    print("   ✗ ADVERTENCIA: Los datasets no tienen los mismos clientes\n")

# Verificar consistencia de target_ternario
print("2. Verificando consistencia de target_ternario...")

ternario_query = """
SELECT 
    COUNT(*) as total_registros,
    SUM(CASE WHEN d1.target_ternario = d2.target_ternario 
             AND d1.target_ternario = d3.target_ternario 
             AND d1.target_ternario = d4.target_ternario 
        THEN 1 ELSE 0 END) as consistentes,
    SUM(CASE WHEN d1.target_ternario != d2.target_ternario 
             OR d1.target_ternario != d3.target_ternario 
             OR d1.target_ternario != d4.target_ternario 
        THEN 1 ELSE 0 END) as inconsistentes
FROM dataset_1 d1
INNER JOIN dataset_2 d2 ON d1.numero_de_cliente = d2.numero_de_cliente
INNER JOIN dataset_3 d3 ON d1.numero_de_cliente = d3.numero_de_cliente
INNER JOIN dataset_4 d4 ON d1.numero_de_cliente = d4.numero_de_cliente
"""

result = conn.execute(ternario_query).fetchone()
print(f"   Total registros comparados: {result[0]}")
print(f"   Consistentes: {result[1]} ({result[1]/result[0]*100:.2f}%)")
print(f"   Inconsistentes: {result[2]} ({result[2]/result[0]*100:.2f}%)")

if result[2] == 0:
    print("   ✓ target_ternario es consistente en todos los datasets\n")
else:
    print("   ✗ ADVERTENCIA: Hay inconsistencias en target_ternario\n")

# Verificar consistencia de target_binario
print("3. Verificando consistencia de target_binario...")

binario_query = """
SELECT 
    COUNT(*) as total_registros,
    SUM(CASE WHEN d1.target_binario = d2.target_binario 
             AND d1.target_binario = d3.target_binario 
             AND d1.target_binario = d4.target_binario 
        THEN 1 ELSE 0 END) as consistentes,
    SUM(CASE WHEN d1.target_binario != d2.target_binario 
             OR d1.target_binario != d3.target_binario 
             OR d1.target_binario != d4.target_binario 
        THEN 1 ELSE 0 END) as inconsistentes
FROM dataset_1 d1
INNER JOIN dataset_2 d2 ON d1.numero_de_cliente = d2.numero_de_cliente
INNER JOIN dataset_3 d3 ON d1.numero_de_cliente = d3.numero_de_cliente
INNER JOIN dataset_4 d4 ON d1.numero_de_cliente = d4.numero_de_cliente
"""

result = conn.execute(binario_query).fetchone()
print(f"   Total registros comparados: {result[0]}")
print(f"   Consistentes: {result[1]} ({result[1]/result[0]*100:.2f}%)")
print(f"   Inconsistentes: {result[2]} ({result[2]/result[0]*100:.2f}%)")

if result[2] == 0:
    print("   ✓ target_binario es consistente en todos los datasets\n")
else:
    print("   ✗ ADVERTENCIA: Hay inconsistencias en target_binario\n")

# Si hay inconsistencias, mostrar ejemplos
inconsistencias_query = """
SELECT 
    d1.numero_de_cliente,
    d1.target_ternario as ternario_d1,
    d2.target_ternario as ternario_d2,
    d3.target_ternario as ternario_d3,
    d4.target_ternario as ternario_d4,
    d1.target_binario as binario_d1,
    d2.target_binario as binario_d2,
    d3.target_binario as binario_d3,
    d4.target_binario as binario_d4
FROM dataset_1 d1
INNER JOIN dataset_2 d2 ON d1.numero_de_cliente = d2.numero_de_cliente
INNER JOIN dataset_3 d3 ON d1.numero_de_cliente = d3.numero_de_cliente
INNER JOIN dataset_4 d4 ON d1.numero_de_cliente = d4.numero_de_cliente
WHERE d1.target_ternario != d2.target_ternario 
   OR d1.target_ternario != d3.target_ternario 
   OR d1.target_ternario != d4.target_ternario
   OR d1.target_binario != d2.target_binario 
   OR d1.target_binario != d3.target_binario 
   OR d1.target_binario != d4.target_binario
LIMIT 10
"""

inconsistencias = conn.execute(inconsistencias_query).fetchall()

if inconsistencias:
    print("="*80)
    print("EJEMPLOS DE INCONSISTENCIAS (primeros 10)")
    print("="*80)
    for row in inconsistencias:
        print(f"\nCliente: {row[0]}")
        print(f"  target_ternario: D1={row[1]}, D2={row[2]}, D3={row[3]}, D4={row[4]}")
        print(f"  target_binario:  D1={row[5]}, D2={row[6]}, D3={row[7]}, D4={row[8]}")

# Cerrar conexión
conn.close()

print("\n" + "="*80)
print("VERIFICACIÓN COMPLETADA")
print("="*80)