import numpy as np
import json
import os
from scipy.optimize import least_squares, differential_evolution, root
from typing import Sequence, Tuple
from scipy.optimize import minimize

# Datos completos (usados para el ajuste y para el bootstrapping)
Q_LIST = np.array([0.690, 1.856, 2.796, 3.620, 5.004, 6.264])  # Caudales en m³/h
S_LIST = np.array([0.5, 1.408, 2.118, 2.768, 3.844, 4.795])         # Abatimientos en m

Q_USE = Q_LIST[:4] 
S_USE = S_LIST[:4]  

# Modelo: S = L*Q + M*Q**2 + I*Q**n

def residuals(params: Sequence[float]) -> np.ndarray:
    L, M, I, n = params
    return S_USE - (L * Q_USE + M * Q_USE**2 + I * Q_USE**n)

def objective_abs(params: Sequence[float]) -> float:
    return np.sum(np.abs(residuals(params)))

# Función auxiliar para bootstrap con subconjunto

def residuals_sub(params: Sequence[float], Q: np.ndarray, S: np.ndarray) -> np.ndarray:
    L, M, I, n = params
    return S - (L * Q + M * Q**2 + I * Q**n)

# Guardar resultados en JSON

def guardar_resultado_json(nombre_archivo: str,
                           metodo: str,
                           coeficientes: list,
                           valor_objetivo: list,
                           limites: dict):
    nuevo_objeto = {
        "Ecuacion propuesta": {
            "metodo": metodo,
            "coeficientes": coeficientes,
            "valor_funcion_objetivo": valor_objetivo,
            "Limites para L": {
                "inferior": limites['L'][0],
                "superior": limites['L'][1]
            },
            "Limites para M": {
                "inferior": limites['M'][0],
                "superior": limites['M'][1]
            },
            "Limites para I": {
                "inferior": limites['I'][0],
                "superior": limites['I'][1]
            },
            "Limites para n": {
                "inferior": limites['n'][0],
                "superior": limites['n'][1]
            }
        }
    }

    if os.path.exists(nombre_archivo):
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            datos = json.load(f)
    else:
        datos = {}

    base_key = "Ecuacion propuesta"
    key = base_key
    i = 1
    while key in datos:
        key = f"{base_key} #{i}"
        i += 1
    datos[key] = nuevo_objeto[base_key]

    with open(nombre_archivo, 'w', encoding='utf-8') as json_file:
        json.dump(datos, json_file, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    # Configuración inicial y límites
    initial_guess = [0.0001, 0.0001, 0.0001, 2]
    lower_bounds = [-2, -2, -2, 1]
    upper_bounds = [2, 2, 2, 3]
    bounds_ls = (lower_bounds, upper_bounds)
    bounds_de = list(zip(lower_bounds, upper_bounds))

    # Least Squares
    res_ls = least_squares(residuals,
                            x0=initial_guess,
                            jac='3-point',
                            bounds=bounds_ls,
                            method='trf',
                            ftol=1e-15,
                            xtol=1e-15,
                            gtol=1e-15,
                            x_scale=1,
                            loss='linear',
                            f_scale=1)
    params_ls = res_ls.x
    obj_ls = objective_abs(params_ls)
    resid_ls = residuals(params_ls)
    print("Resultado least_squares:", params_ls)
    print("Valor de la función objetivo least_squares:", obj_ls)
    print("Residuos least_squares:", resid_ls.tolist())
    print("--------------------------------------------------------")

    # Differential Evolution
    res_de = differential_evolution(objective_abs,
                                    bounds_de,
                                    strategy='best1bin',
                                    maxiter=2000,
                                    init='sobol',
                                    tol=1e-16,
                                    mutation=(0, 1.99),
                                    recombination=0.5,
                                    rng=200,
                                    polish='trust-constr',
                                    atol=1e-16,
                                    updating='immediate',
                                    workers=1)
    params_de = res_de.x
    obj_de = res_de.fun
    resid_de = residuals(params_de)
    print("Resultado differential_evolution:", params_de)
    print("Valor de la función objetivo differential_evolution:", obj_de)
    print("Residuos differential_evolution:", resid_de.tolist())
    print("--------------------------------------------------------")

    # Root
    root_initial = [0.0001, 0.0001, 0.0001, 1]
    sol_root = root(residuals, x0=root_initial, method='hybr')
    if sol_root.success:
        params_root = sol_root.x
        obj_root = objective_abs(params_root)
        resid_root = residuals(params_root)
        print("Solución con root:", params_root)
        print("Valor de la función objetivo root:", obj_root)
        print("Residuos root:", resid_root.tolist())
    else:
        print("root no encontró solución")
        params_root = None
    print("--------------------------------------------------------")

    # Selección de método
    print("Seleccione el método cuyos parámetros desea conservar:")
    print("1 - least_squares")
    print("2 - differential_evolution")
    print("3 - root")
    seleccion = input("Ingrese el número del método (1, 2 o 3): ")

    if seleccion == '1':
        parametros_finales = params_ls
        metodo_seleccionado = 'least_squares'
    elif seleccion == '2':
        parametros_finales = params_de
        metodo_seleccionado = 'differential_evolution'
    elif seleccion == '3' and params_root is not None:
        parametros_finales = params_root
        metodo_seleccionado = 'root'
    else:
        print("Selección inválida o sin solución root.")
        parametros_finales = None
        metodo_seleccionado = None

    if parametros_finales is not None:
        print(f"\nParámetros seleccionados ({metodo_seleccionado}): {parametros_finales}")

        # Errores relativos punto a punto
        Lf, Mf, If, nf = parametros_finales
        S_pred = Lf * Q_USE + Mf * Q_USE**2 + If * Q_USE**nf
        errores_relativos = ((S_USE - S_pred) / S_USE).tolist()

        # Parámetros de bootstrapping
        n_boot = 10000 # número de réplicas
        sigma_S = np.std(S_USE)  # Desviación estándar para el ruido en S_USE

        # Listas para almacenar resultados
        L_boot, M_boot, I_boot, n_boot_list = [], [], [], []
        
        # Bootstrapping
        for i in range(n_boot):
            S_sim = S_USE + np.random.normal(0, sigma_S, size=S_USE.shape)
            res_b = minimize(
                fun=lambda p: np.sum(np.abs(residuals_sub(p, Q_USE, S_sim))),
                x0=parametros_finales,
                method='Powell'
            )
            Lb, Mb, Ib, nb = res_b.x
            L_boot.append(Lb)
            M_boot.append(Mb)
            I_boot.append(Ib)
            n_boot_list.append(nb)

        # Intervalos de confianza 95%
        L_ci = np.percentile(L_boot, [2.5, 97.5])
        M_ci = np.percentile(M_boot, [2.5, 97.5])
        I_ci = np.percentile(I_boot, [2.5, 97.5])
        n_ci = np.percentile(n_boot_list, [2.5, 97.5])

        print("\nIntervalos de confianza (95 %) por bootstrapping:")
        print(f" L: [{L_ci[0]:.6f}, {L_ci[1]:.6f}]")
        print(f" M: [{M_ci[0]:.6f}, {M_ci[1]:.6f}]")
        print(f" I: [{I_ci[0]:.6f}, {I_ci[1]:.6f}]")
        print(f" n: [{n_ci[0]:.6f}, {n_ci[1]:.6f}]")

        limites_boot = {
            'L': (float(L_ci[0]), float(L_ci[1])),  
            'M': (float(M_ci[0]), float(M_ci[1])),  
            'I': (float(I_ci[0]), float(I_ci[1])),  
            'n': (float(n_ci[0]), float(n_ci[1]))   
        }

        # Guardar resultado
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
