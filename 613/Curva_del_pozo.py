import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Datos medidos
# -----------------------------
Q_LIST = np.array([0.690, 1.856, 2.796, 3.620, 5.004, 6.264])  # Caudales en m³/h
S_LIST = np.array([0.5, 1.408, 2.118, 2.768, 3.844, 4.795])         # Abatimientos en m

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
