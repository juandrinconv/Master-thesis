import numpy as np
import matplotlib.pyplot as plt
import json

with open("resultados_PEB3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Caudales a evaluar
Q_eval = np.array([1, 5, 10, 15, 28.8, 55.4, 77.4, 124.2, 150, 175, 200, 250])

# 1) Funcion de ajuste (cuadrática)
coef_ajuste = data["Funcion de ajuste"]["coeficientes"]
S_ajuste = np.polyval(coef_ajuste, Q_eval)

# 2) Ecuacion de Jacob: S = L*Q + M*Q²
L_j, M_j = data["Ecuacion de Jacob"]["coeficientes"]
print(f"L = {L_j:.6e}, M = {M_j:.6e}")
S_jacob = L_j * Q_eval + M_j * Q_eval**2

# 3) Ecuacion de Rorabaugh: S = L*Q + I*Q^n
L_r, I_r, n_r = data["Ecuacion de Rorabaugh"]["coeficientes"]
S_ror = L_r * Q_eval + I_r * Q_eval**n_r

# 4) Propuesta de ajuste: S = L*Q + M*Q² + I*Q^n
L_p, M_p, I_p, n_p = data["Ecuacion propuesta"]["coeficientes"]
S_prop = L_p * Q_eval + M_p * Q_eval**2 + I_p * Q_eval**n_p

# Puntos medidos
Q_meas = np.array([28.8, 55.4, 77.4, 124.2])
S_meas = np.array([0.62, 1.28, 1.96, 3.57])

# -------------------------
# Graficar
# -------------------------
plt.figure(figsize=(12, 7))

# Datos originales
plt.scatter(Q_meas, S_meas, color='crimson', s=150,
            label="Measured Data", edgecolors='black', facecolors='white', linewidths=3)
plt.axvline(Q_meas[0], color='gray', linestyle='--',
            linewidth=1.5)
plt.axvline(Q_meas[-1], color='gray', linestyle='--',
            linewidth=1.5)

# Curvas con líneas y marcadores distintos
plt.plot(Q_eval, S_ajuste, label="Fitting Function",
         marker='None', linewidth=2, color='red')

plt.plot(Q_eval, S_jacob, label="Jacob Equation",
         marker='s', linestyle='dotted', markersize=8)

plt.plot(Q_eval, S_ror, label="Rorabaugh Equation",
         marker='H', linestyle='dotted', markersize=10, markerfacecolor='none',  markeredgewidth=1.5)

plt.plot(Q_eval, S_prop, label="Rincón Equation",
         marker='x', linestyle='dotted', markersize=8)


# Estilo del gráfico
plt.xlabel('Flow Rate Q (m³/h)', fontsize=12)
plt.ylabel('Drawdown $S_W$ (m)', fontsize=12)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()

string_datos_extrapolados_1 = "Extrapolated Data"
plt.text(0.01, 0.3, string_datos_extrapolados_1,
         rotation=35,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

string_datos_medidos = "Measured Data"
plt.text(0.25, 0.2, string_datos_medidos,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

string_datos_extrapolados_2 = "Extrapolated Data"
plt.text(0.6, 0.2, string_datos_extrapolados_2,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

plt.tight_layout()
plt.show()
