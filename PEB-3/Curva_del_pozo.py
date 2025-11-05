import numpy as np
import matplotlib.pyplot as plt
import json

# -----------------------------
# 1) Datos medidos
# -----------------------------
Q_list = np.array([28.8, 55.4, 77.4, 124.2])  # Caudales en m³/h
S_list = np.array([0.62, 1.28, 1.96, 3.57])   # Abatimientos en m

# -----------------------------
# 2) Funcion de R²
# -----------------------------
def calculate_r2(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - ss_res / ss_tot

# -----------------------------
# 3) Ajuste cuadratico
# -----------------------------
quad_coef = np.polyfit(Q_list, S_list, 2)           # [a, b, c]
S_quad = np.polyval(quad_coef, Q_list)
R2_quad = calculate_r2(S_list, S_quad)

# -----------------------------
# 4) Guardar resultados en JSON
# -----------------------------
payload = {
    "Funcion de ajuste": {
        "best_model": "Cuadratico",
        "coeficientes": quad_coef.tolist()
    }
}

with open("resultados_PEB3.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

# -----------------------------
# 5) Graficar
# -----------------------------
Q_curve = np.linspace(Q_list.min(), Q_list.max(), 300)
S_curve = np.polyval(quad_coef, Q_curve)

plt.figure(figsize=(12, 7))

# Datos originales
plt.plot(Q_list, S_list, 'ro', markersize=10, label='Measured Data', zorder=5)

# Curva cuadratica ajustada
plt.plot(Q_curve, S_curve, 'b-', linewidth=3, label='Quadratic Fitting Function', zorder=4)

# Configuracion del grafico
plt.xlabel('Flow Rate Q (m³/h)', fontsize=12)
plt.ylabel('Drawdown $S_W$ (m)', fontsize=12)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()

# Ecuacion y R²
a, b, c = quad_coef
eq = f"$S_W$ = {a:.6f}Q² + {b:.4f}Q + {c:.4f}"
textstr = f"{eq}\nR² = {R2_quad:.6f}"

plt.text(0.05, 0.2, textstr,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

plt.tight_layout()
plt.show()
