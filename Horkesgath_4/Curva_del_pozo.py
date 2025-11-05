import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

# -----------------------------
# 1) Datos medidos
# -----------------------------
Q_list = np.array([30, 45, 60, 70])  # Caudales en m³/h
S_list = np.array([3.88, 6.35, 10.26, 14.16])   # Abatimientos en m

# -----------------------------
# 2) Modelo exponencial y R²
# -----------------------------
def exp_model(Q, a, b):
    return a * np.exp(b * Q)

def calculate_r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred)**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    return 1 - ss_res / ss_tot

# -----------------------------
# 3) Ajuste del modelo exponencial
# -----------------------------
popt, _ = curve_fit(exp_model, Q_list, S_list, p0=[1, 0.01], maxfev=10000)
a_exp, b_exp = popt
S_pred = exp_model(Q_list, a_exp, b_exp)
R2 = calculate_r2(S_list, S_pred)

# -----------------------------
# 4) Guardar en JSON
# -----------------------------
payload = {
    "Funcion de ajuste": {
        "best_model": "Exponencial",
        "coeficientes": [float(a_exp), float(b_exp)]
    }
}

with open("resultados_PEB3.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

# -----------------------------
# 5) Graficar
# -----------------------------
Q_curve = np.linspace(Q_list.min(), Q_list.max(), 300)
S_curve = exp_model(Q_curve, a_exp, b_exp)

plt.figure(figsize=(12, 7))

# Puntos originales
plt.plot(Q_list, S_list, 'ro', markersize=10, label='Measured Data', zorder=5)

# Curva ajustada
plt.plot(Q_curve, S_curve, 'b-', linewidth=3, label='Exponential Fitting Function', zorder=4)

# Mostrar ecuacion
eq_text = f"$S_W = {a_exp:.6f} \\cdot \\exp({b_exp:.4f}Q)$"
textstr = f"{eq_text}\n$R^2$ = {R2:.6f}"

plt.text(0.05, 0.2, textstr,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

# Configuracion
plt.xlabel("Flow Rate Q (m³/h)", fontsize=12)
plt.ylabel("Drawdown $S_W$ (m)", fontsize=12)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
