import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Datos medidos
# -----------------------------
Q_list = np.array([54.417, 70.542, 100.958, 135.875, 170.583, 209.125])  # Caudales en m³/h
S_list = np.array([4.683, 6.514, 9.578, 13.24, 17.395, 22.325])         # Abatimientos en m

# -----------------------------
# Graficar datos con línea azul y marcadores rojos
# -----------------------------
plt.figure(figsize=(12, 7))

# Línea azul
plt.plot(Q_list, S_list, '-', color='blue', linewidth=2.5, zorder=4)

# Marcadores rojos
plt.plot(Q_list, S_list, 'ro', markersize=10, label='Measured Data', zorder=5)

# Configuración de ejes y estilo
plt.xlabel("Flow Rate Q (m³/h)", fontsize=12)
plt.ylabel("Drawdown $S_W$ (m)", fontsize=12)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
