import numpy as np
import matplotlib.pyplot as plt
import json

with open("resultados_PEB3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Caudales a evaluar
Q_eval = np.array([1, 2, 3, 4, 5, 6, 7, 7.23, 14.65, 28.83, 32.69, 33, 33.5, 33.65])

# 1) Funcion de ajuste (Doble exponencial)
coef_ajuste = data["Funcion de ajuste"]["coeficientes"]
def doble_expo(Q, A1, B1, A2, B2, C):
    return A1 * np.exp(B1 * Q) + A2 * np.exp(B2 * Q) + C
A1, B1, A2, B2, C = coef_ajuste
S_ajuste = doble_expo(Q_eval, A1, B1, A2, B2, C)

# 2) Ecuacion de Jacob: S = L*Q + M*Q²
L_j, M_j = data["Ecuacion de Jacob"]["coeficientes"]
S_jacob = L_j * Q_eval + M_j * Q_eval**2

# 3) Ecuacion de Rorabaugh: S = L*Q + I*Q^n
L_r, I_r, n_r = data["Ecuacion de Rorabaugh"]["coeficientes"]
S_ror = L_r * Q_eval + I_r * Q_eval**n_r

# 4) Propuesta de ajuste: S = L*Q + M*Q² + I*Q^n
L_p, M_p, I_p, n_p = data["Ecuacion propuesta"]["coeficientes"]
S_prop = L_p * Q_eval + M_p * Q_eval**2 + I_p * Q_eval**n_p

# Puntos medidos
Q_meas = np.array([7.23, 14.65, 28.83, 32.69])
S_meas = np.array([1.65, 2.9, 10.2, 32.85])

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
         marker='None', linewidth=2, color='red', markersize=7)

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
plt.legend(
    loc='center left',
    bbox_to_anchor=(-0.005, 0.8),
    fontsize=11
)
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()

string_datos_extrapolados_1 = "Extrapolated Data"
plt.text(0.05, 0.2, string_datos_extrapolados_1,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

string_datos_medidos = "Measured Data"
plt.text(0.5, 0.2, string_datos_medidos,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

string_datos_extrapolados_2 = "Extrapolated Data"
plt.text(0.98, 0.4, string_datos_extrapolados_2,
         rotation=90,
         transform=plt.gca().transAxes,
         fontsize=12,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='azure', alpha=0.8))

plt.tight_layout()
plt.show()
