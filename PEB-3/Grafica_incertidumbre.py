import json
import numpy as np
import matplotlib.pyplot as plt


def annotate_label(x, y, text, text_x, text_y, align='left', color='black', fontsize=10):
    ha = 'left' if align == 'left' else 'right'
    plt.annotate(
        text,
        xy=(x, y),
        xytext=(text_x, text_y),
        textcoords='data',
        ha=ha,
        va='center',
        fontsize=fontsize,
        color=color,
        arrowprops=dict(arrowstyle='->', lw=1, color=color)
    )


# Carga del JSON
with open("resultados_PEB3.json", "r", encoding="utf-8") as f:
    r = json.load(f)

# ======== FIGURA COMPLETA CON 3 SUBPLOTS ========
fig, axs = plt.subplots(1, 3, figsize=(14, 5))
plt.subplots_adjust(wspace=0.35)

# 1. Jacob
plt.sca(axs[0])
axs[0].set_title("Jacob's Equation (1947)", fontsize=12)
coefs = r["Ecuacion de Jacob"]["coeficientes"]
lims = r["Ecuacion de Jacob"]
x = np.array([0, 1])
y = np.array([coefs[0], coefs[1]])
lower = np.array([coefs[0] - lims["Limites para L"]["inferior"],
                 coefs[1] - lims["Limites para M"]["inferior"]])
upper = np.array([lims["Limites para L"]["superior"] - coefs[0],
                 lims["Limites para M"]["superior"] - coefs[1]])
yerr = np.vstack([lower, upper])

plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5,
             markeredgewidth=1.5, markeredgecolor='black',
             markerfacecolor='red', ecolor='blue', lw=1.2)
plt.xticks(x, ["L", "M"], fontsize=12)
plt.xlabel("Coefficients", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.grid(True, alpha=0.3)

# L
annotate_label(
    0, y[0],
    f"{y[0]:.3e}",
    0.1, y[0],
    align='left', color='red')
annotate_label(
    0, y[0]-lower[0],
    f"{(y[0]-lower[0]):.3e}",
    0.1, y[0]-lower[0]-0.0001,
    align='left', color='black')
annotate_label(
    0, y[0]+upper[0],
    f"{(y[0]+upper[0]):.3e}",
    0.1, y[0]+upper[0]+0.0001,
    align='left', color='black')

# M
annotate_label(
    1, y[1],
    f"{y[1]:.3e}",
    0.9, y[1] + 0.003,
    align='right', color='red')
annotate_label(
    1, y[1]-lower[1],
    f"{(y[1]-lower[1]):.3e}",
    0.9, y[1]-lower[1]+0.0001,
    align='right', color='black')
annotate_label(
    1, y[1]+upper[1],
    f"{(y[1]+upper[1]):.3e}",
    0.9, y[1]+upper[1]+0.0075,
    align='right', color='black')

# 2. Rorabaugh
plt.sca(axs[1])
axs[1].set_title("Rorabaugh's Equation (1953)", fontsize=12)
coefs = r["Ecuacion de Rorabaugh"]["coeficientes"]
lims = r["Ecuacion de Rorabaugh"]
x = np.arange(3)
y = np.array(coefs)
lower = np.array([
    coefs[0] - lims["Limites para L"]["inferior"],
    coefs[1] - lims["Limites para M"]["inferior"],
    coefs[2] - lims["Limites para n"]["inferior"]
])
upper = np.array([
    lims["Limites para L"]["superior"] - coefs[0],
    lims["Limites para M"]["superior"] - coefs[1],
    lims["Limites para n"]["superior"] - coefs[2]
])
yerr = np.vstack([lower, upper])

plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5,
             markeredgewidth=1.5, markeredgecolor='black',
             markerfacecolor='red', ecolor='blue', lw=1.2)
plt.xticks(x, ["L", "I", "n"], fontsize=12)
plt.xlabel("Coefficients", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.grid(True, alpha=0.3)

# L
annotate_label(
    0, y[0],
    f"{y[0]:.3e}",
    0.2, y[0]-0.06,
    align='left', color='red')
annotate_label(
    0, y[0]-lower[0],
    f"{(y[0]-lower[0]):.3f}",
    0.2, y[0]-lower[0],
    align='left', color='black')
annotate_label(
    0, y[0]+upper[0],
    f"{(y[0]+upper[0]):.3e}",
    0.2, y[0]+upper[0],
    align='left', color='black')

# I
annotate_label(
    1, y[1],
    f"{y[1]:.3e}",
    1.2, y[1],
    align='left', color='red')
annotate_label(
    1, y[1]-lower[1],
    f"{(y[1]-lower[1]):.3e}",
    1.2, y[1]-lower[1]-0.1,
    align='left', color='black')
annotate_label(
    1, y[1]+upper[1],
    f"{(y[1]+upper[1]):.3f}",
    1.2, y[1]+upper[1],
    align='left', color='black')

# n
annotate_label(
    2, y[2],
    f"{y[2]:.3f}",
    1.85, y[2],
    align='right', color='red')
annotate_label(
    2, y[2]-lower[2],
    f"{(y[2]-lower[2]):.3f}",
    1.85, y[2]-lower[2],
    align='right', color='black')
annotate_label(
    2, y[2]+upper[2],
    f"{(y[2]+upper[2]):.3f}",
    1.85, y[2]+upper[2],
    align='right', color='black')

# 3. Propuesta
plt.sca(axs[2])
axs[2].set_title("Rincón's Equation (2025)", fontsize=13)
coefs = r["Ecuacion propuesta"]["coeficientes"]
lims = r["Ecuacion propuesta"]
x = np.arange(4)
y = np.array(coefs)
lower = np.array([
    coefs[0] - lims["Limites para L"]["inferior"],
    coefs[1] - lims["Limites para M"]["inferior"],
    coefs[2] - lims["Limites para I"]["inferior"],
    coefs[3] - lims["Limites para n"]["inferior"]
])
upper = np.array([
    lims["Limites para L"]["superior"] - coefs[0],
    lims["Limites para M"]["superior"] - coefs[1],
    lims["Limites para I"]["superior"] - coefs[2],
    lims["Limites para n"]["superior"] - coefs[3]
])
yerr = np.vstack([lower, upper])

plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5,
             markeredgewidth=1.5, markeredgecolor='black',
             markerfacecolor='red', ecolor='blue', lw=1.2)
plt.xticks(x, ["L", "M", "I", "n"], fontsize=12)
plt.xlabel("Coefficients", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.grid(True, alpha=0.3)

# L
annotate_label(
    0, y[0],
    f"{y[0]:.3e}",
    0.9, y[0]+0.05,
    align='right', color='red',
    fontsize=8)
annotate_label(
    0, y[0]-lower[0],
    f"{(y[0]-lower[0]):.3e}",
    0.9, y[0]-lower[0]-0.05,
    align='right', color='black',
    fontsize=8)
annotate_label(
    0, y[0]+upper[0],
    f"{(y[0]+upper[0]):.3e}",
    0.9, y[0]+upper[0]+0.3,
    align='right', color='black',
    fontsize=8)

# M
annotate_label(
    1, y[1],
    f"{y[1]:.3e}",
    1.2, y[1] + 0.05,
    align='left', color='red',
    fontsize=8)
annotate_label(
    1, y[1]-lower[1],
    f"{(y[1]-lower[1]):.3e}",
    1.2, y[1]-lower[1]-0.06,
    align='left', color='black',
    fontsize=8)
annotate_label(
    1, y[1]+upper[1],
    f"{(y[1]+upper[1]):.3e}",
    1.2, y[1]+upper[1]+0.3,
    align='left', color='black',
    fontsize=8)

# I
annotate_label(
    2, y[2],
    f"{y[2]:.3e}",
    2.3, y[2] + 0.1,
    align='left', color='red',
    fontsize=8)
annotate_label(
    2, y[2]-lower[2],
    f"{(y[2]-lower[2]):.3e}",
    2.3, y[2]-lower[2]-0.05,
    align='left', color='black',
    fontsize=8)
annotate_label(
    2, y[2]+upper[2],
    f"{(y[2]+upper[2]):.3e}",
    2.3, y[2]+upper[2]+0.3,
    align='left', color='black',
    fontsize=8)

# n
annotate_label(
    3, y[3],
    f"{y[3]:.3f}",
    2.8, y[3] - 0.1,
    align='right', color='red',
    fontsize=8)
annotate_label(
    3, y[3]-lower[3],
    f"{(y[3]-lower[3]):.3f}",
    2.8, y[3]-lower[3]-0.18,
    align='right', color='black',
    fontsize=8)
annotate_label(
    3, y[3]+upper[3],
    f"{(y[3]+upper[3]):.3f}",
    2.8, y[3]+upper[3]+0.0001,
    align='right', color='black',
    fontsize=8)

plt.show()
