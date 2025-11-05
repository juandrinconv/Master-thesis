import numpy as np
import json
import os
from scipy.optimize import least_squares, differential_evolution, root
from typing import Sequence, Tuple
from scipy.optimize import minimize

# Datos completos
Q_LIST = np.array([0.177, 0.602, 1.333, 2.203, 3.293])  # Caudales en m³/h
S_LIST = np.array([0.344, 1.033, 2.530, 4.566, 7.223])         # Abatimientos en m

# Datos para el ajuste de Rorabaugh (usando los 3 primeros puntos) y para el bootstrapping
Q_USE = Q_LIST[[0, 1, 2]]
S_USE = S_LIST[[0, 1, 2]]

# Ecuacion: S = L*Q + M*Q^n

def residuals_rorabaugh(params: Sequence[float]) -> np.ndarray:
    L, M, n = params
    return S_USE - (L * Q_USE + M * Q_USE**n)

# Objetivo común: suma de residuos absolutos
def objective_abs(params: Sequence[float]) -> float:
    return np.sum(np.abs(residuals_rorabaugh(params)))

# Least Squares (vector residual) usa residuals

def solve_rorabaugh_ls(
    x0: Sequence[float],
    bounds: Tuple[Sequence[float], Sequence[float]],
    ftol: float = 1e-15,
    xtol: float = 1e-15,
    gtol: float = 1e-15
):
    return least_squares(
        residuals_rorabaugh,
        x0,
        jac='3-point',
        bounds=bounds,
        method='trf',
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        x_scale=1,
        loss='linear',
        f_scale=1
    )

# Differential Evolution usando objetivo de residuos absolutos

def solve_rorabaugh_de(
    bounds: Sequence[Tuple[float, float]],
    maxiter: int = 2000,
    tol: float = 1e-16,
    mutation: Tuple[float, float] = (0, 1.99),
    recombination: float = 0.5,
    rng_seed: int = 200,
    polish: str = 'trust-constr',
    atol: float = 1e-16
):
    return differential_evolution(
        objective_abs,
        bounds,
        strategy='best1bin',
        maxiter=maxiter,
        init='sobol',
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        rng=rng_seed,
        polish=polish,
        atol=atol,
        updating='immediate',
        workers=1
    )

# Root buscando ceros en residuos

def solve_rorabaugh_root(
    x0: Sequence[float],
    method: str = 'hybr'
):
    return root(residuals_rorabaugh, x0, method=method)

# Guardar en JSON

def guardar_resultado_json(nombre_archivo, metodo, coeficientes, valor_objetivo, limites):
    nuevo_objeto = {
        "Ecuacion de Rorabaugh": {
            "metodo": metodo,
            "coeficientes": coeficientes,
            "valor_funcion_objetivo": valor_objetivo,
            "Limites para L": {"inferior": limites['L'][0], "superior": limites['L'][1]},
            "Limites para M": {"inferior": limites['M'][0], "superior": limites['M'][1]},
            "Limites para n": {"inferior": limites['n'][0], "superior": limites['n'][1]}
        }
    }

    if os.path.exists(nombre_archivo):
        with open(nombre_archivo, 'r') as f:
            datos = json.load(f)
    else:
        datos = {}

    base_key = "Ecuacion de Rorabaugh"
    key = base_key
    i = 1
    while key in datos:
        key = f"{base_key} #{i}"
        i += 1

    datos[key] = nuevo_objeto[base_key]

    with open(nombre_archivo, 'w', encoding='utf-8') as json_file:
        json.dump(datos, json_file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    initial_guess = [0.0001, 0.0001, 2]
    lower_bounds = [-2.5, -2.5, 1]
    upper_bounds = [2.5, 2.5, 10]
    bounds_ls = (lower_bounds, upper_bounds)
    bounds_de = list(zip(lower_bounds, upper_bounds))

    # Least Squares
    res_ls = solve_rorabaugh_ls(initial_guess, bounds_ls)
    params_ls = res_ls.x
    obj_ls = objective_abs(params_ls)
    resid_ls = residuals_rorabaugh(params_ls)
    print("\nResultado least_squares:", params_ls)
    print("Valor de la funcion objetivo (sum abs residuos) least_squares:", obj_ls)
    print("Residuos least_squares:", resid_ls.tolist())
    print("--------------------------------------------------------")

    # Differential Evolution
    res_de = solve_rorabaugh_de(bounds_de)
    params_de = res_de.x
    obj_de = res_de.fun
    resid_de = residuals_rorabaugh(params_de)
    print("Resultado differential_evolution:", params_de)
    print("Valor de la funcion objetivo differential_evolution:", obj_de)
    print("Residuos differential_evolution:", resid_de.tolist())
    print("--------------------------------------------------------")

    # Root
    root_initial = [0.0001, 0.0001, 1.5]
    sol_root = solve_rorabaugh_root(root_initial)
    if sol_root.success:
        params_root = sol_root.x
        obj_root = objective_abs(params_root)
        resid_root = residuals_rorabaugh(params_root)
        print("Solucion con root:", params_root)
        print("Valor de la funcion objetivo root:", obj_root)
        print("Residuos root:", resid_root.tolist())
    else:
        print("root no encontro solucion")
        params_root = None
    print("--------------------------------------------------------")

    # Seleccion
    print("Seleccione el metodo cuyos parámetros desea conservar:")
    print("1 - least_squares")
    print("2 - differential_evolution")
    print("3 - root")

    seleccion = input("Ingrese el número del metodo (1, 2 o 3): ")

    if seleccion == '1':
        parametros_finales = params_ls
        metodo_seleccionado = 'least_squares'
    elif seleccion == '2':
        parametros_finales = params_de
        metodo_seleccionado = 'differential_evolution'
    elif seleccion == '3':
        if params_root is not None:
            parametros_finales = params_root
            metodo_seleccionado = 'root'
        else:
            print("No se puede seleccionar root porque no encontro solucion.")
            parametros_finales = None
            metodo_seleccionado = None
    else:
        print("Seleccion inválida.")
        parametros_finales = None
        metodo_seleccionado = None

    if parametros_finales is not None:
        print(
            f"\nParámetros seleccionados ({metodo_seleccionado}): {parametros_finales}")

        # Errores relativos punto a punto (3 valores)
        L, M, n = parametros_finales
        S_pred = L * Q_USE + M * Q_USE**n
        errores_relativos = ((S_USE - S_pred) / S_USE).tolist()

        # Parámetros para bootstrapping
        n_boot = 10000           # número de replicas
        sigma_S = np.std(S_USE)  # desviacion estándar para el ruido en S_USE

        # Listas para almacenar resultados
        L_boot, M_boot, n_boot_list = [], [], []

        def residuals_sub(params, Q, S):
            L, M, n = params
            return S - (L * Q + M * Q**n)
        
        # Bootstrapping
        for i in range(n_boot):
            S_sim = S_USE + np.random.normal(0, sigma_S, size=S_USE.shape)
            res_b = minimize(
                fun=lambda p: np.sum(np.abs(residuals_sub(p, Q_USE, S_sim))),
                x0=parametros_finales,
                method='Powell'
            )
            Lb, Mb, nb = res_b.x
            L_boot.append(Lb)
            M_boot.append(Mb)
            n_boot_list.append(nb)

        L_ci = np.percentile(L_boot, [2.5, 97.5])
        M_ci = np.percentile(M_boot, [2.5, 97.5])
        n_ci = np.percentile(n_boot_list, [2.5, 97.5])

        print("\nIntervalos de confianza (95 %) por bootstrapping:")
        print(f" L: [{L_ci[0]:.6f}, {L_ci[1]:.6f}]")
        print(f" M: [{M_ci[0]:.6f}, {M_ci[1]:.6f}]")
        print(f" n: [{n_ci[0]:.6f}, {n_ci[1]:.6f}]")

        limites_boot = {
            'L': (float(L_ci[0]), float(L_ci[1])),
            'M': (float(M_ci[0]), float(M_ci[1])),
            'n': (float(n_ci[0]), float(n_ci[1]))
        }

        guardar_resultado_json(
            nombre_archivo="resultados_PEB3.json",
            metodo=metodo_seleccionado,
            coeficientes=parametros_finales.tolist(),
            valor_objetivo=errores_relativos,
            limites=limites_boot
        )
        print("Resultado guardado exitosamente en 'resultados_PEB3.json'.")
    else:
        print("No se seleccionaron parámetros finales.")
