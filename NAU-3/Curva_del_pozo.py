import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Datos medidos
# -----------------------------
Q_LIST = np.array([22.713, 45.425, 90.850, 113.562, 124.919])  # Caudales en m³/h
S_LIST = np.array([0.762, 1.981, 6.767, 10.638, 14.173])         # Abatimientos en m

# -----------------------------
# Graficar datos con línea azul y marcadores rojos
# -----------------------------
plt.figure(figsize=(12, 7))

# Línea azul
plt.plot(Q_LIST, S_LIST, '-', color='blue', linewidth=2.5, zorder=4)

# Marcadores rojos
plt.plot(Q_LIST, S_LIST, 'ro', markersize=10, label='Measured Data', zorder=5)

# Configuración de ejes y estilo
plt.xlabel("Flow Rate Q (m³/h)", fontsize=12)
plt.ylabel("Drawdown $S_W$ (m)", fontsize=12)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
