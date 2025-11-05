import json
from pathlib import Path
import numpy as np
import pandas as pd

JSON_PATH = Path("resultados_PEB3.json")
OUT_XLSX = Path("tabla_comparativa_PEB3.xlsx")

# 1) Cargo el JSON de resultados
with open(JSON_PATH, "r", encoding="utf-8") as f:
    resultados = json.load(f)

# 2) Datos medidos (Q y S)
Q_LIST = np.array([0.177, 0.602, 1.333, 2.203, 3.293])  # Caudales en m³/h
S_LIST = np.array([0.344, 1.033, 2.530, 4.566, 7.223])         # Abatimientos en m

# 3) Funciones
def S_jacob(Q, coefs):
    L, M = coefs
    return L * Q + M * Q**2

def S_rorabaugh(Q, coefs):
    L, I, n = coefs
    return L * Q + I * Q**n

def S_propuesta(Q, coefs):
    L, M, I, n = coefs
    return L * Q + M * Q**2 + I * Q**n

# 4) Extraigo coeficientes desde el JSON (utiliza las mismas claves que tu JSON)
coefs_jacob = resultados["Ecuacion de Jacob"]["coeficientes"]
coefs_rorab = resultados["Ecuacion de Rorabaugh"]["coeficientes"]
coefs_prop  = resultados["Ecuacion propuesta"]["coeficientes"]

# 5) Construyo la tabla EXACTA con las columnas originales en ese orden
tabla = pd.DataFrame({
    "Q (m³/h)": Q_LIST,
    "S medido (m)": S_LIST
})

tabla["S Jacob (m)"]      = S_jacob(Q_LIST, coefs_jacob)
tabla["S Rorabaugh (m)"]  = S_rorabaugh(Q_LIST, coefs_rorab)
tabla["S Propuesta (m)"]  = S_propuesta(Q_LIST, coefs_prop)

tabla["Error % Jacob"]     = np.abs(tabla["S Jacob (m)"]     - tabla["S medido (m)"]) / tabla["S medido (m)"] * 100
tabla["Error % Rorabaugh"] = np.abs(tabla["S Rorabaugh (m)"] - tabla["S medido (m)"]) / tabla["S medido (m)"] * 100
tabla["Error % Propuesta"] = np.abs(tabla["S Propuesta (m)"] - tabla["S medido (m)"]) / tabla["S medido (m)"] * 100

# Aseguro el orden exacto (igual al código original)
cols_order = ["Q (m³/h)", "S medido (m)", "S Jacob (m)", "S Rorabaugh (m)", "S Propuesta (m)",
              "Error % Jacob", "Error % Rorabaugh", "Error % Propuesta"]
tabla = tabla[cols_order]

# 6) Guardar sólo a Excel
tabla.to_excel(OUT_XLSX, index=False)
