import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

# -----------------------------
# 1) Datos medidos
# -----------------------------
Q_list = np.array([7.23, 14.65, 28.83, 32.69])  # Caudales en m³/h
S_list = np.array([1.65, 2.9, 10.2, 32.85])       # Abatimientos en m

# -----------------------------
# 2) Modelo: doble exponencial
# -----------------------------
def doble_expo(Q, A1, B1, A2, B2, C):
    return A1 * np.exp(B1 * Q) + A2 * np.exp(B2 * Q) + C

# -----------------------------
# 3) Ajuste del modelo doble exponencial y R²
# -----------------------------
p0 = [1e-4, 0.01, 1e-5, 0.005, 1.0]  # estimaciones iniciales para A1, B1, A2, B2, C
bounds = (0, np.inf)                  # parámetros positivos
popt, _ = curve_fit(doble_expo, Q_list, S_list, p0=p0, bounds=bounds, maxfev=20000)
A1, B1, A2, B2, C = popt

# Prediccion y cálculo de R²
S_pred = doble_expo(Q_list, *popt)
ss_res = np.sum((S_list - S_pred)**2)
ss_tot = np.sum((S_list - np.mean(S_list))**2)
R2 = 1 - ss_res / ss_tot

# -----------------------------
# 4) Guardar en JSON
# -----------------------------
payload = {
    "Funcion de ajuste": {
        "best_model": "Doble Exponencial",
        "coeficientes": [
            float(A1), float(B1), float(A2), float(B2), float(C)
        ]
    }
}
with open("resultados_PEB3.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

# -----------------------------
# 5) Graficar
# -----------------------------
Q_curve = np.linspace(Q_list.min(), Q_list.max(), 300)
S_curve = doble_expo(Q_curve, *popt)

plt.figure(figsize=(12, 7))

# Puntos originales
plt.plot(Q_list, S_list, 'ro', markersize=10, label='Measured Data', zorder=5)

# Curva ajustada
plt.plot(Q_curve, S_curve, 'b-', linewidth=3, label='Double Exponential Fitting Function', zorder=4)

# Mostrar ecuacion y R²
eq_text = (
    f'$S_W = {A1:.6e}e^{{{B1:.4f}Q}} + {A2:.6e}e^{{{B2:.4f}Q}} + {C:.4f}$'
)
textstr = f"{eq_text}\n$R^2 = {R2:.6f}$"

plt.text(0.05, 0.2, textstr,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

# Configuracion de ejes y estilo
plt.xlabel("Flow Rate Q (m³/h)", fontsize=12)
plt.ylabel("Drawdown $S_W$ (m)", fontsize=12)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
