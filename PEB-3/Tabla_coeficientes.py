import json
import pandas as pd

# Cargar el archivo JSON
with open('resultados_PEB3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extraer coeficientes de cada método
coef_jacob = data.get("Ecuacion de Jacob", {}).get("coeficientes", [])
coef_rorab = data.get("Ecuacion de Rorabaugh", {}).get("coeficientes", [])
coef_prop  = data.get("Ecuacion propuesta", {}).get("coeficientes", [])

# Definir nombres de parámetros por método
names_jacob  = ['L', 'M']
names_rorab  = ['L', 'I', 'n']
names_prop   = ['L', 'M', 'I', 'n']

# Lista maestra de todos los parámetros para el índice de la tabla
param_index = ['L', 'M', 'I', 'n']

def align_coefs(coefs, names):
    """Alinea una lista de coeficientes con los nombres dados, rellenando '-' si no existe."""
    aligned = {param: '-' for param in param_index}
    for i, coef in enumerate(coefs):
        if i < len(names):
            aligned[names[i]] = coef
    return aligned

# Alinear coeficientes
aligned_jacob = align_coefs(coef_jacob, names_jacob)
aligned_rorab = align_coefs(coef_rorab, names_rorab)
aligned_prop  = align_coefs(coef_prop,  names_prop)

# Construir DataFrame
df = pd.DataFrame({
    'Jacob': aligned_jacob,
    'Rorabaugh': aligned_rorab,
    'Propuesta de esta tesis': aligned_prop
}, index=param_index)


df.to_excel('tabla_coeficientes.xlsx', index=True)
