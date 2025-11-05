import json
import pandas as pd

# 1) Cargo el JSON de resultados
with open("resultados_PEB3.json", "r", encoding="utf-8") as f:
    resultados = json.load(f)

# 2) Defino los parámetros (filas) y metodologías (columnas)
parametros = ['L', 'M', 'I', 'n']
metodos = {
    'Jacob (1947)':             'Ecuacion de Jacob',
    'Rorabaugh (1953)':         'Ecuacion de Rorabaugh',
    'Propuesta de esta tesis':  'Ecuacion propuesta'
}

# 3) Mapeo interno de cada parámetro al nombre de su límite en el JSON
json_limit_map = {
    'Jacob (1947)': {
        'L': 'L',
        'M': 'M',
        'I': None,
        'n': None
    },
    'Rorabaugh (1953)': {
        'L': 'L',
        'I': 'M',   # En el JSON 'Limites para M' corresponde a I
        'n': 'n',
        'M': None
    },
    'Propuesta de esta tesis': {
        'L': 'L',
        'M': 'M',
        'I': 'I',
        'n': 'n'
    }
}

# 4) Creo DataFrame con MultiIndex en columnas
cols = pd.MultiIndex.from_product(
    [metodos.keys(), ['Límite inferior', 'Límite superior']],
    names=['Metodología', '']
)
tabla = pd.DataFrame(index=parametros, columns=cols, dtype=float)

# 5) Relleno la tabla usando el mapeo
for metodo_nombre, json_key in metodos.items():
    info = resultados[json_key]
    for p in parametros:
        jl = json_limit_map[metodo_nombre].get(p)
        if jl:
            li = info[f"Limites para {jl}"]['inferior']
            ls = info[f"Limites para {jl}"]['superior']
            tabla.loc[p, (metodo_nombre, 'Límite inferior')] = li
            tabla.loc[p, (metodo_nombre, 'Límite superior')] = ls
        else:
            # Parámetro no aplicable: dejar NaN
            tabla.loc[p, (metodo_nombre, 'Límite inferior')] = pd.NA
            tabla.loc[p, (metodo_nombre, 'Límite superior')] = pd.NA

# 6) Exporto a Excel
output_path = "tabla_limites_metodos.xlsx"
tabla.to_excel(output_path, index=True)
