import numpy as np
from scipy.optimize import least_squares, minimize
import json
import os

# Todos los pares de puntos
Q_LIST = np.array([0.690, 1.856, 2.796, 3.620, 5.004, 6.264])  # Caudales en m³/h
S_LIST = np.array([0.5, 1.408, 2.118, 2.768, 3.844, 4.795])         # Abatimientos en m

# Usar solo los dos primeros pares para el ajuste
Q_USE = Q_LIST[:2]
S_USE = S_LIST[:2]

# Modelo para ajuste lineal + cuadrático: S = L*Q + M*Q^2
def modelo(params: np.ndarray, Q: np.ndarray) -> np.ndarray:
    L, M = params
    return L * Q + M * Q**2

# Residual para bootstrap con abs
def residuals_sub(params: np.ndarray, Q: np.ndarray, S: np.ndarray) -> np.ndarray:
    return S - modelo(params, Q)

# --- Ajuste inicial con least_squares ---
x0 = [0.0001, 0.0001]  # Valores iniciales para L y M
lb = [-2, -2]          # Límites inferiores
ub = [2, 2]            # Límites superiores

res = least_squares(
    fun=lambda p: residuals_sub(p, Q_USE, S_USE),
    x0=x0,
    bounds=(lb, ub),
    jac='2-point'
)
L_opt, M_opt = res.x

print("Ajuste original con dos puntos (least_squares):")
print(f" L = {L_opt:.6f}")
print(f" M = {M_opt:.6f}")

# --- Bootstrapping ---
n_boot = 10000  # Número de replicas
sigma = np.std(S_USE)  # desviacion estándar para el ruido en S

L_boot = []
M_boot = []

for i in range(n_boot):
    S_sim = S_USE + np.random.normal(0, sigma, size=S_USE.shape)
    res_b = minimize(
        fun=lambda p: np.sum(np.abs(residuals_sub(p, Q_USE, S_sim))),
        x0=res.x,
        method='Powell'
    )
    Lb, Mb = res_b.x
    L_boot.append(Lb)
    M_boot.append(Mb)

# Calcular intervalos de confianza del 95%
L_ci = np.percentile(L_boot, [2.5, 97.5])
M_ci = np.percentile(M_boot, [2.5, 97.5])

print("\nIntervalos de confianza del 95% (bootstrapping):")
print(f" L: [{L_ci[0]:.6f}, {L_ci[1]:.6f}]")
print(f" M: [{M_ci[0]:.6f}, {M_ci[1]:.6f}]")

# Evaluar la funcion objetivo (residuos) con los coeficientes optimos
residuos_opt = residuals_sub(res.x, Q_USE, S_USE)
print("\nResiduos punto a punto con los coeficientes encontrados:", residuos_opt.tolist())

# --- Guardar resultados en JSON ---
json_file_path = 'resultados_PEB3.json'
nuevo_resultado = {
    "metodo": "least_squares",
    "coeficientes": [float(L_opt), float(M_opt)],
    "valor_funcion_objetivo": [float(r) for r in residuos_opt],
    "Limites para L": {"inferior": float(L_ci[0]), "superior": float(L_ci[1])},
    "Limites para M": {"inferior": float(M_ci[0]), "superior": float(M_ci[1])}
}

# Leer o inicializar
if os.path.exists(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {}
else:
    data = {}

# Actualizar seccion
data["Ecuacion de Jacob"] = nuevo_resultado

# Volver a guardar
with open(json_file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nSe ha actualizado '{json_file_path}' con la seccion 'Ecuacion de Jacob'.")
