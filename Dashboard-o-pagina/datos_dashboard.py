"""Backend de datos del dashboard.

Todas las cifras que muestra el tablero se calculan aquí, a partir de los mismos
archivos y del mismo código que usa el equipo de Código:

- ``wc_predictor.py`` (carga del histórico, Elo, forma reciente y promedios con
  *shrinkage*) se importa directamente desde la carpeta del código.
- Las funciones de modelación de ``Analisis.ipynb`` (secciones 2, 4 y 7) viven dentro
  del notebook, no en un módulo importable, por eso se reproducen aquí con la misma
  lógica y los mismos nombres. ``tabla_verificacion()`` comprueba que el tablero
  obtiene exactamente las métricas reportadas por el notebook.

Lo que el notebook no calcula (análisis exploratorio, tasa de aciertos, intervalos
bootstrap, calibración, descomposición de la brecha con el mercado y el simulador)
está marcado como "complemento del dashboard" en cada función.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson, skellam
from sklearn.metrics import log_loss, mean_absolute_error


AQUI = Path(__file__).resolve().parent
RUTA_CODIGO = Path(os.environ.get("RUTA_CODIGO", AQUI.parent / "Codigo" / "proyecto_mod_8")).resolve()
ARCHIVO_HISTORICO = RUTA_CODIGO / "E0_consolidado.csv"
ARCHIVO_VARIABLES = RUTA_CODIGO / "premier_training_data.csv"


@contextlib.contextmanager
def _en_directorio(ruta: Path):
    anterior = Path.cwd()
    os.chdir(ruta)
    try:
        yield
    finally:
        os.chdir(anterior)


def _importar_wc_predictor():
    """Importa wc_predictor tal cual lo dejó el equipo.

    Al importarse, el módulo lee "E0_consolidado.csv" con ruta relativa e imprime el
    ranking Elo, por eso se importa desde su carpeta y con la salida silenciada.
    """
    if str(RUTA_CODIGO) not in sys.path:
        sys.path.insert(0, str(RUTA_CODIGO))
    with _en_directorio(RUTA_CODIGO), contextlib.redirect_stdout(io.StringIO()):
        import wc_predictor
    return wc_predictor


wc_predictor = _importar_wc_predictor()


FECHA_INICIO = pd.Timestamp("2019-08-01")
FECHA_VALIDACION = pd.Timestamp("2024-08-01")
FECHA_PRUEBA = pd.Timestamp("2025-08-01")

K_SHRINKAGE = 10
N_FORMA = 10
DECAY_FORMA = 0.85
ESCALA_ELO = 400

CLAVES = ["Date", "HomeTeam", "AwayTeam"]
OBJETIVOS = ["home_goals", "away_goals"]
CUOTAS = ["AvgH", "AvgD", "AvgA"]
CUOTAS_CIERRE = ["AvgCH", "AvgCD", "AvgCA"]
PROBS = ["P_home", "P_draw", "P_away"]

GRUPOS = {
    "base_home": ["elo_diff", "gf_home", "ga_away"],
    "base_away": ["elo_diff", "gf_away", "ga_home"],
    "forma_home": ["form_gf_home", "form_ga_away"],
    "forma_away": ["form_gf_away", "form_ga_home"],
    "tiros_home": ["shots_for_home", "shots_against_away"],
    "tiros_away": ["shots_for_away", "shots_against_home"],
    "sot_home": ["sot_for_home", "sot_against_away"],
    "sot_away": ["sot_for_away", "sot_against_home"],
}

ESPECIFICACIONES = {
    "M0_Base": (GRUPOS["base_home"], GRUPOS["base_away"]),
    "M1_Forma": (
        GRUPOS["base_home"] + GRUPOS["forma_home"],
        GRUPOS["base_away"] + GRUPOS["forma_away"],
    ),
    "M2_Tiros": (
        GRUPOS["base_home"] + GRUPOS["tiros_home"],
        GRUPOS["base_away"] + GRUPOS["tiros_away"],
    ),
    "M3_SOT": (
        GRUPOS["base_home"] + GRUPOS["sot_home"],
        GRUPOS["base_away"] + GRUPOS["sot_away"],
    ),
    "M4_Completo": (
        GRUPOS["base_home"] + GRUPOS["forma_home"] + GRUPOS["tiros_home"] + GRUPOS["sot_home"],
        GRUPOS["base_away"] + GRUPOS["forma_away"] + GRUPOS["tiros_away"] + GRUPOS["sot_away"],
    ),
}

# Modelo que usa el simulador: el mejor en prueba y el más parsimonioso (3 variables por ecuación).
MODELO_SIMULADOR = "M0_Base"

NOMBRES = {
    "Uniforme": "Azar · 1/3 por resultado",
    "Ingenua": "Referencia ingenua",
    "M0_Base": "M0 · Base",
    "M1_Forma": "M1 · + Forma",
    "M2_Tiros": "M2 · + Tiros",
    "M3_SOT": "M3 · + Tiros a puerta",
    "M4_Completo": "M4 · Completo",
    "Mercado_apertura": "Mercado · Apertura",
    "Mercado_cierre": "Mercado · Cierre",
}

# Cifras publicadas en Analisis.ipynb (secciones 5, 7 y 8) y en el reporte técnico.
REPORTADO_NOTEBOOK = {
    ("Validación", "M0_Base"): 0.983690,
    ("Validación", "M1_Forma"): 0.982334,
    ("Validación", "M2_Tiros"): 0.977038,
    ("Validación", "M3_SOT"): 0.977225,
    ("Validación", "M4_Completo"): 0.975296,
    ("Validación", "Mercado_apertura"): 0.970552,
    ("Prueba", "M0_Base"): 1.030556,
    ("Prueba", "M2_Tiros"): 1.035184,
    ("Prueba", "M4_Completo"): 1.034434,
    ("Prueba", "Mercado_apertura"): 1.020000,
}
REPORTADO_EJEMPLO_M0 = {"lambda_home": 1.566, "lambda_away": 1.202,
                        "P_home": 0.4576, "P_draw": 0.2497, "P_away": 0.2927}

SEMILLA = 2026
N_BOOTSTRAP = 10_000


# ── 2. Funciones de Analisis.ipynb (secciones 2 y 4), misma lógica ─────────────
def crear_variables_partido(df_pre, fecha, home, away, elo_home, elo_away):
    """Predictores construidos sólo con partidos anteriores a `fecha` (notebook §2)."""
    fecha = pd.Timestamp(fecha)
    if not (df_pre["date"] < fecha).all():
        raise ValueError("df_pre contiene información de la fecha objetivo o posterior")

    stats_h = wc_predictor.season_stats(df_pre, home, fecha, k=K_SHRINKAGE)
    stats_a = wc_predictor.season_stats(df_pre, away, fecha, k=K_SHRINKAGE)
    form_h = wc_predictor.recent_form(df_pre, home, n=N_FORMA, decay=DECAY_FORMA)
    form_a = wc_predictor.recent_form(df_pre, away, n=N_FORMA, decay=DECAY_FORMA)

    return {
        "elo_home": elo_home, "elo_away": elo_away,
        "elo_diff": elo_home - elo_away,
        "gf_home": stats_h["gf_avg"], "ga_home": stats_h["ga_avg"],
        "gf_away": stats_a["gf_avg"], "ga_away": stats_a["ga_avg"],
        "form_gf_home": form_h["gf"], "form_ga_home": form_h["ga"],
        "form_gf_away": form_a["gf"], "form_ga_away": form_a["ga"],
        "shots_for_home": form_h["shots_for"],
        "shots_against_home": form_h["shots_against"],
        "shots_for_away": form_a["shots_for"],
        "shots_against_away": form_a["shots_against"],
        "sot_for_home": form_h["sot_for"],
        "sot_against_home": form_h["sot_against"],
        "sot_for_away": form_a["sot_for"],
        "sot_against_away": form_a["sot_against"],
    }


def preparar_X(datos, columnas, modelo=None):
    """Prepara regresores, incluida la escala de Elo y el intercepto (notebook §4)."""
    X = datos.loc[:, columnas].copy().astype(float)
    X["elo_diff"] = X["elo_diff"] / ESCALA_ELO
    X = sm.add_constant(X, has_constant="add")
    if modelo is not None:
        X = X.loc[:, modelo.model.exog_names]
    return X


def entrenar_modelo(datos, columnas_home, columnas_away):
    """Una regresión Poisson para goles locales y otra para visitantes (notebook §4)."""
    home = sm.GLM(datos["home_goals"], preparar_X(datos, columnas_home),
                  family=sm.families.Poisson()).fit()
    away = sm.GLM(datos["away_goals"], preparar_X(datos, columnas_away),
                  family=sm.families.Poisson()).fit()
    return {"home": home, "away": away,
            "columnas_home": columnas_home, "columnas_away": columnas_away}


def probabilidades_1x2(lambda_home, lambda_away):
    """[victoria local, empate, victoria visitante] vía Skellam (notebook §4)."""
    lh, la = np.asarray(lambda_home), np.asarray(lambda_away)
    probs = np.column_stack((
        skellam.sf(0, lh, la),
        skellam.pmf(0, lh, la),
        skellam.cdf(-1, lh, la),
    ))
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Las probabilidades 1X2 no son válidas")
    return probs


def resultados_observados(datos):
    """Codifica 0=local, 1=empate y 2=visitante (notebook §4)."""
    gh = datos["home_goals"].to_numpy()
    ga = datos["away_goals"].to_numpy()
    return np.where(gh > ga, 0, np.where(gh == ga, 1, 2))


def predecir_con_modelo(modelo, datos):
    """Goles esperados y probabilidades para un conjunto de encuentros (notebook §4)."""
    lh = np.asarray(modelo["home"].predict(
        preparar_X(datos, modelo["columnas_home"], modelo["home"])), dtype=float)
    la = np.asarray(modelo["away"].predict(
        preparar_X(datos, modelo["columnas_away"], modelo["away"])), dtype=float)
    resultado = datos[CLAVES + [c for c in OBJETIVOS if c in datos]].reset_index(drop=True).copy()
    resultado["lambda_home"] = lh
    resultado["lambda_away"] = la
    resultado[PROBS] = probabilidades_1x2(lh, la)
    return resultado


def evaluar_predicciones(pred):
    """MAE de goles y Log-Loss 1X2 (notebook §4)."""
    mae_h = mean_absolute_error(pred["home_goals"], pred["lambda_home"])
    mae_a = mean_absolute_error(pred["away_goals"], pred["lambda_away"])
    return {
        "MAE_local": mae_h,
        "MAE_visitante": mae_a,
        "MAE_promedio": (mae_h + mae_a) / 2,
        "LogLoss_1X2": log_loss(resultados_observados(pred), pred[PROBS].to_numpy(), labels=[0, 1, 2]),
    }


def probabilidades_mercado(datos, historico, columnas=CUOTAS):
    """Cuotas decimales -> probabilidad implícita normalizada (notebook §7).

    q_j = 1/cuota_j ; p_j = q_j / (q_1 + q_X + q_2). La normalización reparte el
    margen de la casa de apuestas de forma proporcional.
    """
    x = datos[CLAVES].merge(historico[CLAVES + columnas], on=CLAVES, how="left", validate="one_to_one")
    brutas = 1.0 / x[columnas].to_numpy(dtype=float)
    return brutas / brutas.sum(axis=1, keepdims=True), (brutas.sum(axis=1).mean() - 1.0) * 100


# ── 3. Carga de datos ─────────────────────────────────────────────────────────
def _etiqueta_temporada(anio: int) -> str:
    return f"{anio}/{str(anio + 1)[-2:]}"


@lru_cache(maxsize=None)
def cargar_historico() -> pd.DataFrame:
    """Base consolidada del equipo (9 450 partidos) con temporada y resultado."""
    df = pd.read_csv(ARCHIVO_HISTORICO)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    # La temporada va del 1 de agosto al 31 de julio, igual que en wc_predictor.season_stats.
    df["temporada"] = np.where(df["Date"].dt.month >= 8, df["Date"].dt.year, df["Date"].dt.year - 1)
    df["resultado"] = np.where(df["FTHG"] > df["FTAG"], 0, np.where(df["FTHG"] == df["FTAG"], 1, 2))
    return df


@lru_cache(maxsize=None)
def cargar_base_modelacion() -> pd.DataFrame:
    """premier_training_data.csv, la misma caché que usa el notebook (§3)."""
    td = pd.read_csv(ARCHIVO_VARIABLES)
    td["Date"] = pd.to_datetime(td["Date"])
    return td.sort_values("Date", kind="stable").reset_index(drop=True)


def particiones():
    td = cargar_base_modelacion()
    train = td.loc[td["Date"] < FECHA_VALIDACION].copy()
    val = td.loc[(td["Date"] >= FECHA_VALIDACION) & (td["Date"] < FECHA_PRUEBA)].copy()
    test = td.loc[td["Date"] >= FECHA_PRUEBA].copy()
    return train, val, test


# ── 4. Análisis exploratorio (complemento del dashboard) ──────────────────────
def resumen_general():
    df = cargar_historico()
    completas = df[df["temporada"] <= 2025]
    return {
        "partidos": len(df),
        "temporadas": df["temporada"].nunique(),
        "fecha_min": df["Date"].min(),
        "fecha_max": df["Date"].max(),
        "partidos_completas": len(completas),
        "pct_local": (completas["resultado"] == 0).mean(),
        "pct_empate": (completas["resultado"] == 1).mean(),
        "pct_visita": (completas["resultado"] == 2).mean(),
        "goles_partido": (completas["FTHG"] + completas["FTAG"]).mean(),
        "goles_local": completas["FTHG"].mean(),
        "goles_visita": completas["FTAG"].mean(),
    }


def resumen_temporadas() -> pd.DataFrame:
    """Resultados y goles por temporada completa (2001/02–2025/26)."""
    df = cargar_historico()
    df = df[df["temporada"] <= 2025]
    t = df.groupby("temporada").agg(
        partidos=("resultado", "size"),
        pct_local=("resultado", lambda s: (s == 0).mean()),
        pct_empate=("resultado", lambda s: (s == 1).mean()),
        pct_visita=("resultado", lambda s: (s == 2).mean()),
        goles_local=("FTHG", "mean"),
        goles_visita=("FTAG", "mean"),
    ).reset_index()
    t["etiqueta"] = t["temporada"].map(_etiqueta_temporada)
    t["ventaja_local_pp"] = (t["pct_local"] - t["pct_visita"]) * 100
    return t


def resultado_del_favorito():
    """Qué pasa con el favorito de Bet365 (cuota más baja entre local y visitante).

    Se usa Bet365 porque es la única casa con cuotas en 24 temporadas (2002/03 en
    adelante); las cuotas promedio (Avg) sólo existen desde 2019/20.
    """
    df = cargar_historico().dropna(subset=["B365H", "B365D", "B365A"])
    local_favorito = df["B365H"] <= df["B365A"]
    fav = np.where(local_favorito, 0, 2)
    res = df["resultado"].to_numpy()
    return {
        "partidos": len(df),
        "gana_favorito": (res == fav).mean(),
        "empate": (res == 1).mean(),
        "gana_no_favorito": ((res != fav) & (res != 1)).mean(),
        "local_es_favorito": local_favorito.mean(),
        "temporada_min": _etiqueta_temporada(int(df["temporada"].min())),
    }


def resultado_por_elo(n_grupos: int = 10) -> pd.DataFrame:
    """Frecuencia de cada resultado por decil de diferencia Elo previa al partido."""
    td = cargar_base_modelacion().copy()
    td["resultado"] = resultados_observados(td)
    td["grupo"] = pd.qcut(td["elo_diff"], n_grupos, labels=False)
    t = td.groupby("grupo").agg(
        partidos=("resultado", "size"),
        elo_min=("elo_diff", "min"),
        elo_max=("elo_diff", "max"),
        elo_mediana=("elo_diff", "median"),
        pct_local=("resultado", lambda s: (s == 0).mean()),
        pct_empate=("resultado", lambda s: (s == 1).mean()),
        pct_visita=("resultado", lambda s: (s == 2).mean()),
    ).reset_index()
    return t


def ventaja_local_en_elo(tabla: pd.DataFrame) -> float:
    """Diferencia Elo en la que P(local) = P(visitante), interpolando entre deciles.

    Si el local necesita ser X puntos más débil para empatar las probabilidades de
    ganar, jugar en casa "vale" aproximadamente X puntos Elo.
    """
    x = tabla["elo_mediana"].to_numpy()
    d = (tabla["pct_local"] - tabla["pct_visita"]).to_numpy()
    for i in range(len(d) - 1):
        if d[i] < 0 <= d[i + 1]:
            return float(-(x[i] + (0 - d[i]) * (x[i + 1] - x[i]) / (d[i + 1] - d[i])))
    return float("nan")


def calibracion_historica_mercado(ancho: float = 0.05, minimo: int = 50) -> pd.DataFrame:
    """Probabilidad implícita (Bet365) vs frecuencia observada, 2002/03–2026/27."""
    df = cargar_historico().dropna(subset=["B365H", "B365D", "B365A"])
    brutas = 1 / df[["B365H", "B365D", "B365A"]].to_numpy(float)
    q = brutas / brutas.sum(axis=1, keepdims=True)
    ocurrio = np.eye(3)[df["resultado"].to_numpy()]
    return _tabla_calibracion(q.ravel(), ocurrio.ravel(), ancho, minimo)


def _tabla_calibracion(p, o, ancho, minimo):
    bordes = np.arange(0, 1 + ancho, ancho)
    grupo = pd.cut(p, bordes, include_lowest=True)
    t = (pd.DataFrame({"p": p, "o": o, "g": grupo})
         .groupby("g", observed=True)
         .agg(n=("o", "size"), prob_predicha=("p", "mean"), frecuencia=("o", "mean"))
         .reset_index(drop=True))
    return t[t["n"] >= minimo].reset_index(drop=True)


def ajuste_poisson_goles(max_goles: int = 6) -> pd.DataFrame:
    """Distribución observada de goles vs Poisson con la misma media (temporadas completas)."""
    df = cargar_historico()
    df = df[df["temporada"] <= 2025]
    filas = []
    for col, lado in [("FTHG", "Local"), ("FTAG", "Visitante")]:
        g = df[col].astype(int)
        lam = g.mean()
        for k in range(max_goles + 1):
            if k < max_goles:
                obs, teo = (g == k).mean(), poisson.pmf(k, lam)
            else:  # la última categoría acumula "6 o más"
                obs, teo = (g >= k).mean(), poisson.sf(k - 1, lam)
            filas.append({"lado": lado, "goles": k, "etiqueta": f"{k}+" if k == max_goles else str(k),
                          "observado": obs, "poisson": teo, "media": lam, "varianza": g.var()})
    return pd.DataFrame(filas)


# ── 5. Modelación y evaluación ────────────────────────────────────────────────
@lru_cache(maxsize=None)
def modelos_entrenados():
    """M0–M4 ajustados en entrenamiento (2019/20–2023/24), como en el notebook §5."""
    train, _, _ = particiones()
    return {nombre: entrenar_modelo(train, *cols) for nombre, cols in ESPECIFICACIONES.items()}


@lru_cache(maxsize=None)
def probabilidades_por_conjunto():
    """Probabilidades 1X2 de cada predictor en validación y prueba.

    Incluye las referencias sin información (azar y referencia ingenua del notebook §6),
    los cinco modelos y el mercado (apertura, y cierre como referencia adicional).
    """
    train, val, test = particiones()
    historico = cargar_historico()
    modelos = modelos_entrenados()
    lh0, la0 = float(train["home_goals"].mean()), float(train["away_goals"].mean())
    salida = {}
    for nombre_conjunto, datos in [("Validación", val), ("Prueba", test)]:
        n = len(datos)
        P = {
            "Uniforme": np.full((n, 3), 1 / 3),
            "Ingenua": probabilidades_1x2(np.full(n, lh0), np.full(n, la0)),
        }
        predicciones = {}
        for nombre, modelo in modelos.items():
            predicciones[nombre] = predecir_con_modelo(modelo, datos)
            P[nombre] = predicciones[nombre][PROBS].to_numpy()
        P["Mercado_apertura"], margen = probabilidades_mercado(datos, historico, CUOTAS)
        P["Mercado_cierre"], _ = probabilidades_mercado(datos, historico, CUOTAS_CIERRE)
        salida[nombre_conjunto] = {
            "datos": datos.reset_index(drop=True),
            "y": resultados_observados(datos),
            "P": P,
            "predicciones": predicciones,
            "margen": margen,
        }
    return salida


def _logloss_por_partido(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1.0))


def tabla_metricas() -> pd.DataFrame:
    """LogLoss, probabilidad media asignada al resultado real, aciertos y MAE."""
    filas = []
    for conjunto, c in probabilidades_por_conjunto().items():
        y = c["y"]
        for nombre, P in c["P"].items():
            ll = log_loss(y, P, labels=[0, 1, 2])
            fila = {
                "conjunto": conjunto, "predictor": nombre, "nombre": NOMBRES[nombre],
                "partidos": len(y), "logloss": ll,
                # exp(-LogLoss) = media geométrica de la probabilidad dada al resultado real.
                "prob_resultado_real": float(np.exp(-ll)),
                "aciertos": float((P.argmax(axis=1) == y).mean()),
                "p_empate_media": float(P[:, 1].mean()),
                "mae_promedio": np.nan,
            }
            if nombre in c["predicciones"]:
                fila["mae_promedio"] = evaluar_predicciones(c["predicciones"][nombre])["MAE_promedio"]
            filas.append(fila)
    return pd.DataFrame(filas)


def fraccion_de_mejora(metricas: pd.DataFrame, conjunto="Prueba", modelo="M0_Base") -> float:
    """Qué parte de la mejora del mercado sobre la referencia ingenua logra el modelo."""
    m = metricas[metricas["conjunto"] == conjunto].set_index("predictor")["logloss"]
    return float((m["Ingenua"] - m[modelo]) / (m["Ingenua"] - m["Mercado_apertura"]))


def _bootstrap(diferencias, semilla=SEMILLA, B=N_BOOTSTRAP):
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(diferencias), size=(B, len(diferencias)), dtype=np.int32)
    medias = diferencias[idx].mean(axis=1)
    return float(diferencias.mean()), float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


@lru_cache(maxsize=None)
def tabla_bootstrap() -> pd.DataFrame:
    """IC 95 % bootstrap (remuestreo de partidos) de diferencias de LogLoss.

    Complemento del dashboard: el notebook reporta las diferencias pero no su
    incertidumbre. Diferencia = A − B; negativa significa que A fue mejor.
    """
    c = probabilidades_por_conjunto()
    comparaciones = [
        ("Prueba", "M0_Base", "Ingenua"),
        ("Prueba", "M0_Base", "Mercado_apertura"),
        ("Prueba", "M4_Completo", "M0_Base"),
        ("Prueba", "M4_Completo", "Mercado_apertura"),
        ("Validación", "M4_Completo", "M0_Base"),
        ("Validación", "M0_Base", "Mercado_apertura"),
        ("Validación + prueba", "M0_Base", "Mercado_apertura"),
    ]
    filas = []
    for conjunto, a, b in comparaciones:
        partes = ["Validación", "Prueba"] if conjunto == "Validación + prueba" else [conjunto]
        dif = np.concatenate([
            _logloss_por_partido(c[p]["y"], c[p]["P"][a]) - _logloss_por_partido(c[p]["y"], c[p]["P"][b])
            for p in partes
        ])
        media, lo, hi = _bootstrap(dif)
        filas.append({"conjunto": conjunto, "a": a, "b": b, "partidos": len(dif),
                      "diferencia": media, "ic_inf": lo, "ic_sup": hi,
                      "significativa": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(filas)


def delta_contra_m0(metricas: pd.DataFrame) -> pd.DataFrame:
    """ΔLogLoss de cada predictor respecto a M0, en validación y en prueba."""
    m = metricas.pivot(index="predictor", columns="conjunto", values="logloss")
    orden = ["M1_Forma", "M2_Tiros", "M3_SOT", "M4_Completo", "Mercado_apertura"]
    d = pd.DataFrame({
        "predictor": orden,
        "nombre": [NOMBRES[p] for p in orden],
        "validacion": [m.loc[p, "Validación"] - m.loc["M0_Base", "Validación"] for p in orden],
        "prueba": [m.loc[p, "Prueba"] - m.loc["M0_Base", "Prueba"] for p in orden],
    })
    return d


def brecha_por_resultado() -> pd.DataFrame:
    """Aporte de cada resultado real a la diferencia de LogLoss M0 − mercado.

    Complemento del dashboard. Se combinan validación y prueba (799 partidos):
    M0 no se eligió con ninguno de los dos periodos, así que ambos son fuera de muestra.
    Positivo = el mercado asignó más probabilidad a lo que ocurrió.
    """
    c = probabilidades_por_conjunto()
    y = np.concatenate([c[p]["y"] for p in ("Validación", "Prueba")])
    Pm = np.vstack([c[p]["P"]["M0_Base"] for p in ("Validación", "Prueba")])
    Pk = np.vstack([c[p]["P"]["Mercado_apertura"] for p in ("Validación", "Prueba")])
    dif = _logloss_por_partido(y, Pm) - _logloss_por_partido(y, Pk)
    filas = []
    for k, nombre in enumerate(["Victoria local", "Empate", "Victoria visitante"]):
        s = y == k
        filas.append({"resultado": nombre, "partidos": int(s.sum()),
                      "aporte": float(dif[s].sum() / len(y)),
                      "p_modelo": float(Pm[s, k].mean()), "p_mercado": float(Pk[s, k].mean())})
    t = pd.DataFrame(filas)
    t.attrs["brecha_total"] = float(dif.mean())
    t.attrs["partidos"] = len(y)
    return t


def calibracion_modelo_vs_mercado(ancho: float = 0.1, minimo: int = 30) -> pd.DataFrame:
    """Calibración de M0 y del mercado de apertura (validación + prueba)."""
    c = probabilidades_por_conjunto()
    y = np.concatenate([c[p]["y"] for p in ("Validación", "Prueba")])
    ocurrio = np.eye(3)[y].ravel()
    tablas = []
    for nombre in ("M0_Base", "Mercado_apertura"):
        P = np.vstack([c[p]["P"][nombre] for p in ("Validación", "Prueba")])
        t = _tabla_calibracion(P.ravel(), ocurrio, ancho, minimo)
        t["predictor"] = nombre
        tablas.append(t)
    return pd.concat(tablas, ignore_index=True)


def efectos_estandarizados(modelo="M0_Base") -> pd.DataFrame:
    """Cambio % en goles esperados por +1 desviación estándar de cada variable.

    exp(β·DE) − 1, con su IC 95 %. Permite comparar variables con escalas distintas
    (Elo/400, goles por partido, tiros). Complemento del dashboard.
    """
    train, _, _ = particiones()
    m = modelos_entrenados()[modelo]
    etiquetas = {
          "elo_diff": "Diferencia Elo",
          "gf_home": "Goles a favor · local",
          "ga_away": "Goles recibidos · visitante",
          "gf_away": "Goles a favor · visitante",
          "ga_home": "Goles recibidos · local",
    }
    filas = []
    for lado, nombre_lado in [("home", "Goles del local"), ("away", "Goles del visitante")]:
        ajuste = m[lado]
        X = preparar_X(train, m[f"columnas_{lado}"])
        de = X.drop(columns="const").std()
        ic = ajuste.conf_int()
        for v in de.index:
            filas.append({
                "ecuacion": nombre_lado, "variable": v, "etiqueta": etiquetas.get(v, v),
                "coef": ajuste.params[v], "p_valor": ajuste.pvalues[v],
                "efecto": np.exp(ajuste.params[v] * de[v]) - 1,
                "ic_inf": np.exp(ic.loc[v, 0] * de[v]) - 1,
                "ic_sup": np.exp(ic.loc[v, 1] * de[v]) - 1,
            })
    return pd.DataFrame(filas)


def dispersion_pearson(modelo="M0_Base"):
    """χ² de Pearson / grados de libertad: ≈ 1 si la varianza condicional ≈ media (Poisson)."""
    m = modelos_entrenados()[modelo]
    return {lado: float(m[lado].pearson_chi2 / m[lado].df_resid) for lado in ("home", "away")}


def sensibilidad_sin_publico() -> dict:
    """M0 reentrenado sin los partidos a puerta cerrada (jun-2020 a may-2021).

    Complemento del dashboard: responde si la temporada sin público, incluida en el
    entrenamiento, explica la brecha con el mercado. No sustituye al modelo del equipo.
    """
    train, val, test = particiones()
    cerrado = (train["Date"] >= "2020-06-17") & (train["Date"] <= "2021-05-23")
    salida = {"partidos_excluidos": int(cerrado.sum())}
    for etiqueta, datos in [("original", train), ("sin_publico", train[~cerrado])]:
        m = entrenar_modelo(datos, *ESPECIFICACIONES["M0_Base"])
        for conjunto, d in [("val", val), ("test", test)]:
            pred = predecir_con_modelo(m, d)
            salida[f"{etiqueta}_{conjunto}_logloss"] = evaluar_predicciones(pred)["LogLoss_1X2"]
            salida[f"{etiqueta}_{conjunto}_p_local"] = float(pred["P_home"].mean())
    return salida


def comparacion_partido_a_partido(conjunto="Prueba") -> pd.DataFrame:
    c = probabilidades_por_conjunto()[conjunto]
    d = c["datos"][CLAVES].copy()
    d["p_local_modelo"] = c["P"]["M0_Base"][:, 0]
    d["p_local_mercado"] = c["P"]["Mercado_apertura"][:, 0]
    d["resultado"] = np.array(["Local", "Empate", "Visitante"])[c["y"]]
    return d


# ── 6. Simulador (complemento del dashboard) ──────────────────────────────────
def datos_simulador(max_goles: int = 5):
    """Probabilidades de M0 para todos los cruces de la temporada 2026/27.

    Usa la misma construcción de variables que el notebook (§9, predecir_partido) con
    corte al día siguiente del último partido disponible.
    """
    historial = wc_predictor.load_history(ARCHIVO_HISTORICO)
    historial["date"] = pd.to_datetime(historial["date"])
    fecha_corte = historial["date"].max() + pd.Timedelta(days=1)
    df_pre = historial.loc[historial["date"] < fecha_corte].copy()
    temporada_actual = int(fecha_corte.year if fecha_corte.month >= 8 else fecha_corte.year - 1)
    actuales = df_pre[df_pre["date"] >= pd.Timestamp(temporada_actual, 8, 1)]
    equipos = sorted(set(actuales["home_team"]) | set(actuales["away_team"]))
    elo = wc_predictor.build_elo(df_pre)
    modelo = modelos_entrenados()[MODELO_SIMULADOR]

    filas = [
        {"Date": fecha_corte, "HomeTeam": h, "AwayTeam": a,
         **crear_variables_partido(df_pre, fecha_corte, h, a,
                                   elo.get(h, wc_predictor.ELO_INIT), elo.get(a, wc_predictor.ELO_INIT))}
        for h in equipos for a in equipos if h != a
    ]
    pred = predecir_con_modelo(modelo, pd.DataFrame(filas))
    goles = np.arange(11)
    partidos = []
    for r in pred.itertuples(index=False):
        matriz = np.outer(poisson.pmf(goles, r.lambda_home), poisson.pmf(goles, r.lambda_away))
        gh, ga = np.unravel_index(matriz.argmax(), matriz.shape)
        partidos.append({
            "local": r.HomeTeam, "visita": r.AwayTeam,
            "lambda_local": round(float(r.lambda_home), 3), "lambda_visita": round(float(r.lambda_away), 3),
            "p_local": round(float(r.P_home), 4), "p_empate": round(float(r.P_draw), 4),
            "p_visita": round(float(r.P_away), 4),
            "marcador": f"{gh}-{ga}", "p_marcador": round(float(matriz[gh, ga]), 4),
            "matriz": [[round(float(matriz[i, j]), 4) for j in range(max_goles + 1)]
                       for i in range(max_goles + 1)],
        })

    stats = []
    for e in equipos:
        s = wc_predictor.season_stats(df_pre, e, fecha_corte, k=K_SHRINKAGE)
        n_actual = int(((actuales["home_team"] == e) | (actuales["away_team"] == e)).sum())
        stats.append({"equipo": e, "elo": round(float(elo.get(e, wc_predictor.ELO_INIT)), 1),
                      "gf": round(float(s["gf_avg"]), 2), "ga": round(float(s["ga_avg"]), 2),
                      "partidos_temporada": n_actual})
    return {"fecha_corte": fecha_corte, "temporada": _etiqueta_temporada(temporada_actual),
            "equipos": equipos, "partidos": partidos, "stats": stats}


# ── 7. Calidad de datos y verificación ────────────────────────────────────────
def auditoria_datos() -> pd.DataFrame:
    """Revisión de calidad de E0_consolidado.csv (complemento del dashboard)."""
    df = cargar_historico()
    por_temp = df.groupby("temporada").size()
    incompletas = [f"{_etiqueta_temporada(t)} ({n})" for t, n in por_temp.items() if n < 380 and t < 2026]
    ftr = np.where(df["FTHG"] > df["FTAG"], "H", np.where(df["FTHG"] < df["FTAG"], "A", "D"))
    malos_tiros = df[(df["HST"] > df["HS"]) | (df["AST"] > df["AS"])]
    detalle_tiros = "; ".join(
        f"{r.HomeTeam}–{r.AwayTeam} {r.Date:%Y-%m-%d}" for r in malos_tiros.itertuples()) or "—"
    goles_sin_tiro = int((df["FTHG"] > df["HST"]).sum() + (df["FTAG"] > df["AST"]).sum())
    td = cargar_base_modelacion()
    completas_2019 = df[(df["Date"] >= FECHA_INICIO)]
    excluidos = len(completas_2019) - len(td)
    filas = [
        ("Registros", f"{len(df):,} partidos, 33 columnas", "Base completa del equipo, 18-ago-2001 a 14-sep-2026"),
        ("Duplicados (fecha, local, visitante)", f"{df.duplicated(CLAVES).sum()}", "Sin duplicados"),
        ("Resultado (FTR) incongruente con los goles", f"{(ftr != df['FTR']).sum()}", "Sin incongruencias"),
        ("Temporadas incompletas", ", ".join(incompletas) or "—",
         "Faltan 45 partidos en cada una; afecta sólo al Elo histórico, no al periodo de modelación"),
        ("Tiros a puerta mayores que tiros", f"{len(malos_tiros)}", f"Error de la fuente: {detalle_tiros}; no se corrigió"),
        ("Goles mayores que tiros a puerta", f"{goles_sin_tiro}", "Plausible (autogoles); no es error"),
        ("Cuotas Bet365", "faltan en 2001/02", "Se usan desde 2002/03 para el análisis histórico del mercado"),
        ("Cuotas promedio de apertura y cierre", "sólo desde 2019/20", "Por eso la comparación modelo–mercado es 2024/25–2026/27"),
        ("Goles esperados (xG)", "sólo 2026/27", "No se usan: cobertura insuficiente"),
        ("Partidos excluidos de la base de modelación", f"{excluidos}",
         "Equipos sin historial previo de tiros en la base (Brentford 2021, Nott'm Forest 2022, Luton 2023, Coventry 2026)"),
    ]
    return pd.DataFrame(filas, columns=["Revisión", "Resultado", "Comentario"])


def tabla_verificacion(metricas: pd.DataFrame, simulador) -> pd.DataFrame:
    """Compara cifras del dashboard contra las publicadas por el notebook/reporte."""
    filas = []
    m = metricas.set_index(["conjunto", "predictor"])["logloss"]
    for (conjunto, pred), valor in REPORTADO_NOTEBOOK.items():
        calc = m.loc[(conjunto, pred)]
        filas.append({"Cifra": f"LogLoss {NOMBRES[pred]} ({conjunto.lower()})",
                      "Notebook": f"{valor:.6f}", "Dashboard": f"{calc:.6f}",
                      "Coincide": abs(calc - valor) < 5e-7})
    ej = next(p for p in simulador["partidos"] if p["local"] == "Arsenal" and p["visita"] == "Man City")
    comparar = [("λ Arsenal (M0)", "lambda_home", ej["lambda_local"], 3),
                ("λ Man City (M0)", "lambda_away", ej["lambda_visita"], 3),
                ("P(victoria Arsenal) M0", "P_home", ej["p_local"], 4),
                ("P(empate) M0", "P_draw", ej["p_empate"], 4),
                ("P(victoria Man City) M0", "P_away", ej["p_visita"], 4)]
    for nombre, clave, valor, dec in comparar:
        ref = REPORTADO_EJEMPLO_M0[clave]
        filas.append({"Cifra": f"Ejemplo del reporte: {nombre}", "Notebook": f"{ref:.{dec}f}",
                      "Dashboard": f"{valor:.{dec}f}", "Coincide": abs(valor - ref) < 0.6 * 10 ** -dec})
    return pd.DataFrame(filas)


@lru_cache(maxsize=None)
def preparar_todo():
    """Calcula todo lo que usa el dashboard en una sola pasada."""
    metricas = tabla_metricas()
    elo_tabla = resultado_por_elo()
    simulador = datos_simulador()
    return {
        "general": resumen_general(),
        "temporadas": resumen_temporadas(),
        "favorito": resultado_del_favorito(),
        "elo": elo_tabla,
        "ventaja_elo": ventaja_local_en_elo(elo_tabla),
        "calibracion_mercado": calibracion_historica_mercado(),
        "poisson": ajuste_poisson_goles(),
        "metricas": metricas,
        "fraccion_mejora": fraccion_de_mejora(metricas),
        "fraccion_mejora_val": fraccion_de_mejora(metricas, "Validación"),
        "bootstrap": tabla_bootstrap(),
        "delta_m0": delta_contra_m0(metricas),
        "brecha": brecha_por_resultado(),
        "calibracion_modelos": calibracion_modelo_vs_mercado(),
        "efectos": efectos_estandarizados(),
        "dispersion": dispersion_pearson(),
        "sensibilidad": sensibilidad_sin_publico(),
        "partido_a_partido": comparacion_partido_a_partido(),
        "simulador": simulador,
        "auditoria": auditoria_datos(),
        "verificacion": tabla_verificacion(metricas, simulador),
        "margen": {k: v["margen"] for k, v in probabilidades_por_conjunto().items()},
        "particiones": {k: len(v) for k, v in zip(("train", "val", "test"), particiones())},
    }


if __name__ == "__main__":
    d = preparar_todo()
    pd.set_option("display.width", 200)
    print(d["metricas"].round(4).to_string())
    print(d["bootstrap"].round(4).to_string())
    print(d["verificacion"].to_string())
    print("Fracción de mejora (prueba, validación):", round(d["fraccion_mejora"], 3), round(d["fraccion_mejora_val"], 3))
    print("Ventaja local en Elo:", round(d["ventaja_elo"], 1))
    print("Simulador:", d["simulador"]["fecha_corte"].date(), len(d["simulador"]["partidos"]), "cruces")
