import json
import numpy as np
import pandas as pd

# 1) Cargo el JSON de resultados
with open("resultados_PEB3.json", "r", encoding="utf-8") as f:
    resultados = json.load(f)

# 2) Defino el vector de caudales
Q = np.array([1, 5, 10, 15, 28.8, 55.4, 77.4, 124.2, 150, 175, 200, 250])

# 3) Calculo S_medido usando la 'Funcion de ajuste' (Cuadrático)
fit = resultados["Funcion de ajuste"]
a, b, c = fit["coeficientes"]
S_medido = a * Q**2 + b * Q + c

# 4) Funciones para cada método
def S_jacob(Q, coefs):
    L, M = coefs
    return L * Q + M * Q**2

def S_rorabaugh(Q, coefs):
    L, I, n = coefs
    return L * Q + I * Q**n

def S_propuesta(Q, coefs):
    L, M, I, n = coefs
    return L * Q + M * Q**2 + I * Q**n

# 5) Construyo el DataFrame
tabla = pd.DataFrame({
    "Q (m³/h)": Q,
    "S medido (m)": S_medido
})

# 6) Abatimientos calculados
tabla["S Jacob (m)"]      = S_jacob(Q, resultados["Ecuacion de Jacob"]["coeficientes"])
tabla["S Rorabaugh (m)"]  = S_rorabaugh(Q, resultados["Ecuacion de Rorabaugh"]["coeficientes"])
tabla["S Propuesta (m)"]  = S_propuesta(Q, resultados["Ecuacion propuesta"]["coeficientes"])

# 7) Error relativo absoluto para cada método
tabla["Error % Jacob"]     = np.abs(tabla["S Jacob (m)"]     - tabla["S medido (m)"]) / tabla["S medido (m)"] * 100
tabla["Error % Rorabaugh"] = np.abs(tabla["S Rorabaugh (m)"] - tabla["S medido (m)"]) / tabla["S medido (m)"] * 100
tabla["Error % Propuesta"] = np.abs(tabla["S Propuesta (m)"] - tabla["S medido (m)"]) / tabla["S medido (m)"] * 100


# Guardar a Excel
tabla.to_excel("tabla_comparativa_PEB3.xlsx", index=False)
