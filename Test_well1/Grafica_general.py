import numpy as np
import matplotlib.pyplot as plt
import json

with open("resultados_PEB3.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Puntos medidos
Q_LIST = np.array([54.417, 70.542, 100.958, 135.875, 170.583, 209.125])  # Caudales en m³/h
S_LIST = np.array([4.683, 6.514, 9.578, 13.24, 17.395, 22.325])   # Abatimientos en m


# 2) Ecuacion de Jacob: S = L*Q + M*Q²
L_j, M_j = data["Ecuacion de Jacob"]["coeficientes"]
S_jacob = L_j * Q_LIST + M_j * Q_LIST**2

# 3) Ecuacion de Rorabaugh: S = L*Q + I*Q^n
L_r, I_r, n_r = data["Ecuacion de Rorabaugh"]["coeficientes"]
S_ror = L_r * Q_LIST + I_r * Q_LIST**n_r

# 4) Propuesta de ajuste: S = L*Q + M*Q² + I*Q^n
L_p, M_p, I_p, n_p = data["Ecuacion propuesta"]["coeficientes"]
S_prop = L_p * Q_LIST + M_p * Q_LIST**2 + I_p * Q_LIST**n_p

# -------------------------
# Graficar
# -------------------------
plt.figure(figsize=(12, 7))

# Datos originales
plt.scatter(Q_LIST, S_LIST, color='crimson', s=150,
            label="Measured Data", edgecolors='black', facecolors='white', linewidths=3)

# Curvas con líneas y marcadores distintos
plt.plot(Q_LIST, S_LIST, label="Fitting Function",
         marker='None', linewidth=2, color='red', markersize=7)

plt.plot(Q_LIST, S_jacob, label="Jacob Equation",
         marker='s', linestyle='dotted', markersize=8)

plt.plot(Q_LIST, S_ror, label="Rorabaugh Equation",
         marker='H', linestyle='dotted', markersize=10, markerfacecolor='none',  markeredgewidth=1.5)

plt.plot(Q_LIST, S_prop, label="Rincón Equation",
         marker='x', linestyle='dotted', markersize=8)


# Estilo del gráfico
plt.xlabel('Flow Rate Q (m³/h)', fontsize=12)
plt.ylabel('Drawdown $S_W$ (m)', fontsize=12)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
