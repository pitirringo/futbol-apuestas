import numpy as np
import pandas as pd

# ── Constantes del modelo ─────────────────────────────────────────────────────


ELO_INIT     = 1500
ELO_SCALE    = 400

# Valores de K por default, estas las tomamos despues de realizar prueba log loss
ELO_K        = 15
SHRINKAGE_K  = 0


# ── Carga de histórico ────────────────────────────────────────────────────────

# Define la función que carga y prepara el histórico de partidos.
# Si no se especifica otra ruta, utiliza E0_consolidado.csv.
def load_history(
    path="E0_consolidado.csv"
):

    # Lee el archivo CSV y lo almacena como un DataFrame de pandas.
    df = pd.read_csv(path)

    # Renombra las variables principales para utilizar nombres homogéneos dentro del predictor.
    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_score",
        "FTAG": "away_score"
    })

    # Convierte la columna de fecha a formato datetime, dayfirst=True indica que el día aparece antes que el mes, mientras que format="mixed" permite distintos formatos de fecha.
    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        format="mixed"
    )

    # Elimina los partidos que no tengan alguna de las variables indispensables: fecha, equipos o marcador final.
    df = df.dropna(subset=[
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score"
    ])

    # Ordena los partidos cronológicamente, reinicia el índice y devuelve el DataFrame preparado.
    return df.sort_values("date").reset_index(drop=True)



def calcular_parametros_liga(df):
    """
    Calcula los promedios históricos de goles y los factores de localía a partir del DataFrame.
    """

    # Calcula el promedio de goles anotados por los equipos locales.
    avg_home_goals = df["home_score"].mean()

    # Calcula el promedio de goles anotados por los equipos visitantes.
    avg_away_goals = df["away_score"].mean()

    # Calcula el promedio general de goles anotados por equipo a partir de los promedios de local y visitante.
    avg_team_goals = (
        avg_home_goals + avg_away_goals
    ) / 2

    # Calcula el factor de localía como la proporción entre los goles promedio del local y el promedio general.
    home_factor = avg_home_goals / avg_team_goals

    # Calcula el factor de visitante como la proporción entre los goles promedio del visitante y el promedio general.
    away_factor = avg_away_goals / avg_team_goals

    # Devuelve los tres parámetros calculados en un diccionario.
    return {
        "avg_team_goals": avg_team_goals,
        "home_factor": home_factor,
        "away_factor": away_factor
    }

# Cargar el histórico
df = load_history()

# Calcular parámetros de la liga: calcula los promedios históricos de goles y los factores de localía a partir del DataFrame.
parametros = calcular_parametros_liga(df)

# Extrae del diccionario el promedio general de goles por equipo.
avg_team_goals = parametros["avg_team_goals"]

# Extrae el factor de localía calculado a partir del histórico.
HOME_FACTOR = parametros["home_factor"]

# Extrae el factor correspondiente a los equipos visitantes.
AWAY_FACTOR = parametros["away_factor"]
# ── Elo ────────────────────────────────────────────────────────────────────────

# Calcula la probabilidad esperada de que el equipo A obtenga un resultado favorable frente al equipo B a partir de la diferencia entre sus ratings Elo.
#Notese que ELO_SCALE la tomamos como 400 como base
def expected_score(r_a, r_b, scale=ELO_SCALE):

    # Transforma la diferencia de ratings en una probabilidad entre 0 y 1. ELO_SCALE controla la sensibilidad de la probabilidad a dicha diferencia.
    return 1 / (1 + 10 ** ((r_b - r_a) / scale))


# Actualiza el rating Elo del equipo A después de observar el resultado del partido.
def update_elo(r_a, r_b, score_a, k=ELO_K, scale=ELO_SCALE):

    # Calcula el resultado esperado del equipo A antes del partido.
    exp_a = expected_score(r_a, r_b, scale=scale)

    # Actualiza el Elo según la diferencia entre el resultado observado
    # (1 = victoria, 0.5 = empate, 0 = derrota) y el resultado esperado.
    # k determina qué tan rápido responde el rating a nueva información, por defecto es 30.
    return r_a + k * (score_a - exp_a)

def build_elo(df, elo_init=ELO_INIT, k=ELO_K, scale=ELO_SCALE):
    """
    Calcula el rating Elo de cada club de la Premier League
    utilizando los resultados históricos.

    Los partidos se procesan cronológicamente.
    Cada equipo comienza con un Elo inicial de 1500.
    """

    # Crea un diccionario vacío donde se almacenará el rating Elo
    # actualizado de cada equipo.
    ratings = {}

    # Ordena los partidos cronológicamente y recorre el histórico
    # partido por partido.
    for _, row in df.sort_values("date").iterrows():

        # Extrae el nombre del equipo local.
        h = row["home_team"]

        # Extrae el nombre del equipo visitante.
        a = row["away_team"]

        # Extrae los goles anotados por el equipo local.
        hs = row["home_score"]

        # Extrae los goles anotados por el equipo visitante.
        as_ = row["away_score"]

        # Elo anterior al partido

        # Obtiene el Elo actual del local, si el equipo todavía no aparece en el diccionario, se le asigna el Elo inicia.
        rh = ratings.get(h, elo_init)

        # Obtiene el Elo actual del visitante, si el equipo todavía no aparece en el diccionario, se le asigna el Elo inicial.
        ra = ratings.get(a, elo_init)

        # Resultado del partido

        # Si el local anota más goles, se asigna 1 al local y 0 al visitante.
        if hs > as_:
            sh, sa = 1, 0

        # Si el visitante anota más goles, se asigna 0 al local y 1 al visitante.
        elif hs < as_:
            sh, sa = 0, 1

        # Si el partido termina empatado, ambos equipos reciben un resultado de 0.5.
        else:
            sh, sa = 0.5, 0.5

        # Actualización de Elo

        # Actualiza el Elo del equipo local comparando su resultado observado con el resultado esperado frente al Elo del visitante.
        ratings[h] = update_elo(rh, ra, sh, k=k, scale=scale)

        # Actualiza el Elo del equipo visitante utilizando los ratings que ambos equipos tenían antes del partido.
        ratings[a] = update_elo(ra, rh, sa, k=k, scale=scale)

    # Devuelve un diccionario con el Elo final de cada equipo después de procesar todos los partidos del histórico.
    return ratings


# ── Forma reciente ────────────────────────────────────────────────────────────

def recent_form(df, team, n=10, decay=0.85):
    """
    Calcula la forma reciente de un equipo utilizando
    sus últimos n partidos.

    Variables:
        gf: goles anotados
        ga: goles recibidos
        shots_for: tiros realizados
        shots_against: tiros concedidos
        sot_for: tiros a puerta realizados
        sot_against: tiros a puerta concedidos

    Los partidos más recientes reciben mayor peso.
    """

    # Seleccionar los últimos n partidos del equipo

    # Filtra todos los partidos en los que el equipo participó,
    # ya sea como local o como visitante.
    tmp = df[
        (df["home_team"] == team) |  #| O
        (df["away_team"] == team)
    ].sort_values("date").tail(n)  # Ordena los partidos cronológicamente y conserva únicamente los n encuentros más recientes.

    # Si no hay partidos disponibles:

    # Comprueba si el equipo no tiene ningún partido disponible dentro del histórico proporcionado.
    if tmp.empty:

        # Devuelve el promedio general de goles como valor inicial y NaN para las estadísticas de tiros que no pueden calcularse.
        return {
            "gf": avg_team_goals,
            "ga": avg_team_goals,
            "shots_for": np.nan,
            "shots_against": np.nan,
            "sot_for": np.nan,
            "sot_against": np.nan
        }

    # Inicializar listas

    # Crea las listas donde se almacenarán los goles anotados y recibidos en cada uno de los partidos seleccionados.
    gf = []
    ga = []

    # Crea las listas donde se almacenarán los tiros realizados y concedidos.
    shots_for = []
    shots_against = []

    # Crea las listas donde se almacenarán los tiros a puerta realizados y concedidos.
    sot_for = []
    sot_against = []

    # Recorrer los partidos

    # Recorre uno por uno los partidos seleccionados.
    for _, row in tmp.iterrows():

        # Comprueba si el equipo analizado jugó el partido como local.
        if row["home_team"] == team:

            # Goles

            # Guarda los goles anotados por el equipo.
            gf.append(row["home_score"])

            # Guarda los goles recibidos por el equipo.
            ga.append(row["away_score"])

            # Tiros

            # Guarda los tiros realizados por el equipo local.
            shots_for.append(row["HS"])

            # Guarda los tiros realizados por el rival y, por tanto, concedidos por el equipo analizado.
            shots_against.append(row["AS"])

            # Tiros a puerta

            # Guarda los tiros a puerta realizados por el equipo local.
            sot_for.append(row["HST"])

            # Guarda los tiros a puerta realizados por el rival.
            sot_against.append(row["AST"])

        # Si el equipo analizado no fue local, entonces jugó como visitante, obtenemos las mismas metricas pero de visitante.
        else:

            # Goles

            # Guarda los goles anotados por el equipo como visitante.
            gf.append(row["away_score"])

            # Guarda los goles recibidos por el equipo como visitante.
            ga.append(row["home_score"])

            # Tiros

            # Guarda los tiros realizados por el equipo visitante.
            shots_for.append(row["AS"])

            # Guarda los tiros realizados por el equipo local y,
            # por tanto, concedidos por el equipo analizado.
            shots_against.append(row["HS"])

            # Tiros a puerta

            # Guarda los tiros a puerta realizados por el equipo visitante.
            sot_for.append(row["AST"])

            # Guarda los tiros a puerta realizados por el equipo local.
            sot_against.append(row["HST"])

    # Pesos geométricos

    # Construye un vector de pesos geométricos utilizando el parámetro decay.
    # El partido más reciente recibe peso 1 y los anteriores reciben pesos progresivamente menores: decay, decay^2, decay^3, etc.
    w = np.array([

        # Calcula el peso correspondiente a cada partido según su posición temporal dentro de la muestra reciente.
        decay ** (len(gf) - 1 - i)

        # Repite el cálculo para todos los partidos disponibles.
        for i in range(len(gf))
    ])

    # Normalizar los pesos

    # Divide cada peso entre la suma total para conseguir que todos los pesos sumen 1.
    w = w / w.sum()

    # Promedios ponderados

    # Devuelve las estadísticas de forma reciente calculadas como promedios ponderados por la antigüedad de cada partido.
    return {

        # Promedio ponderado de goles anotados.
        "gf": float(np.average(gf, weights=w)),

        # Promedio ponderado de goles recibidos.
        "ga": float(np.average(ga, weights=w)),

        # Promedio ponderado de tiros realizados.
        "shots_for": float(
            np.average(shots_for, weights=w)
        ),

        # Promedio ponderado de tiros concedidos.
        "shots_against": float(
            np.average(shots_against, weights=w)
        ),

        # Promedio ponderado de tiros a puerta realizados.
        "sot_for": float(
            np.average(sot_for, weights=w)
        ),

        # Promedio ponderado de tiros a puerta concedidos.
        "sot_against": float(
            np.average(sot_against, weights=w)
        )
    }

def season_stats(df, team, as_of_date, k=SHRINKAGE_K):
    """
    Calcula los goles promedio a favor y en contra
    de un equipo utilizando shrinkage.

    Combina:
        - Estadísticas de la temporada actual.
        - Estadísticas de la temporada anterior.

    k controla el peso de la información anterior.

    Utiliza exclusivamente partidos anteriores
    a la fecha de predicción.
    """

    # Convierte la fecha de predicción a formato datetime de pandas.
    fecha = pd.to_datetime(as_of_date)

    # Evitar utilizar partidos futuros

    # Conserva únicamente los partidos disputados antes de la fecha para la cual se realizará la predicción.
    df_pre = df[df["date"] < fecha].copy()

    # Identificar el inicio de la temporada actual

    # Si la fecha se encuentra entre agosto y diciembre, la temporada comenzó en ese mismo año.
    if fecha.month >= 8:
        season_year = fecha.year

    # Si la fecha se encuentra entre enero y julio, la temporada comenzó durante el año anterior.
    else:
        season_year = fecha.year - 1

    # Construye la fecha correspondiente al 1 de agosto, utilizada como inicio operativo de la temporada actual.
    season_start = pd.Timestamp(
        year=season_year,
        month=8,
        day=1
    )

    # Construye la fecha correspondiente al 1 de agosto del año anterior, es decir, el inicio de la temporada previa.
    previous_start = pd.Timestamp(
        year=season_year - 1,
        month=8,
        day=1
    )

    # Partidos de la temporada actual

    # Selecciona todos los partidos disputados desde el inicio de la temporada actual hasta antes de la fecha de predicción.
    current = df_pre[
        df_pre["date"] >= season_start
    ]

    # De los partidos de la temporada actual, conserva únicamente aquellos en los que participó el equipo analizado.
    current = current[
        (current["home_team"] == team) |
        (current["away_team"] == team)
    ]

    #Partidos de la temporada anterior

    # Selecciona todos los partidos de la liga correspondientes a la temporada inmediatamente anterior.
    previous_league = df_pre[
        (df_pre["date"] >= previous_start) &
        (df_pre["date"] < season_start)
    ]

    # De la temporada anterior, conserva únicamente los partidos disputados por el equipo analizado.
    previous = previous_league[
        (previous_league["home_team"] == team) |
        (previous_league["away_team"] == team)
    ]

    # Función auxiliar para calcular GF/GA

    # Define una función interna que calcula los promedios de goles a favor (GF) y goles en contra (GA) del equipo.
    def calculate_stats(matches):

        # Comprueba si no existen partidos disponibles.
        if matches.empty:

            # Devuelve None para indicar que no es posible calcular estadísticas para ese periodo.
            return None

        # Construye un vector con los goles anotados por el equipo. Si jugó como local utiliza home_score, si jugó como visitante utiliza away_score.
        gf = np.where(
            matches["home_team"] == team,
            matches["home_score"],
            matches["away_score"]
        )

        # Construye un vector con los goles recibidos por el equipo. Si jugó como local utiliza los goles del visitante, pero 
        # si jugó como visitante utiliza los goles del local.
        ga = np.where(
            matches["home_team"] == team,
            matches["away_score"],
            matches["home_score"]
        )

        # Devuelve los promedios de goles a favor y en contra calculados sobre los partidos proporcionados.
        return {
            "gf": float(np.mean(gf)),
            "ga": float(np.mean(ga))
        }

    # Estadísticas de ambas temporadas

    # Calcula los promedios GF y GA del equipo en la temporada actual.
    stats_current = calculate_stats(current)

    # Calcula los promedios GF y GA del equipo en la temporada anterior.
    stats_previous = calculate_stats(previous)

    # Construir el promedio previo


    # Comprueba si el equipo no cuenta con información correspondiente a la temporada anterior.
    if stats_previous is None:

        # Si el equipo no jugó la temporada anterior, utilizar el promedio de goles de esa liga.

        # Comprueba que existan partidos de la Premier League correspondientes a la temporada anterior.
        if not previous_league.empty:

            # Calcula el promedio de goles por equipo y partido de toda la liga durante la temporada anterior.
            league_avg = (
                previous_league["home_score"].sum()
                + previous_league["away_score"].sum()
            ) / (2 * len(previous_league))

        else:

            # Respaldo para temporadas sin datos previos

            # Busca todos los partidos disponibles anteriores al comienzo de la temporada actual.
            historical = df_pre[
                df_pre["date"] < season_start
            ]

            # Comprueba si tampoco existe información histórica anterior al comienzo de la temporada.
            if historical.empty:

                # Detiene la ejecución porque no existe información con la cual construir el promedio previo.
                raise ValueError(
                    "No existe histórico anterior suficiente "
                    "para calcular el promedio previo."
                )

            # Si existe histórico, calcula el promedio de goles por equipo y partido utilizando toda la información anterior.
            league_avg = (
                historical["home_score"].sum()
                + historical["away_score"].sum()
            ) / (2 * len(historical))

        # Utiliza el promedio de la liga como estimación previa de los goles a favor del equipo.
        gf_previo = league_avg

        # Utiliza el mismo promedio de la liga como estimación previa de los goles en contra del equipo.
        ga_previo = league_avg

    else:

        # Si el equipo sí jugó la temporada anterior, utiliza su promedio de goles a favor como información previa.
        gf_previo = stats_previous["gf"]

        # Utiliza su promedio de goles en contra de la temporada
        # anterior como información previa.
        ga_previo = stats_previous["ga"]


    # Aplicar shrinkage


    # Cuenta cuántos partidos ha disputado el equipo durante la temporada actual antes de la fecha de predicción.
    n = len(current)

    # Comprueba si el equipo todavía no ha disputado ningún partido durante la temporada actual.
    if n == 0:

        # Si no existe información de la temporada actual, utiliza directamente las estadísticas previas.
        return {
            "gf_avg": gf_previo,
            "ga_avg": ga_previo
        }

    # Extrae el promedio de goles a favor observado durante la temporada actual.
    gf_actual = stats_current["gf"]

    # Extrae el promedio de goles en contra observado durante la temporada actual.
    ga_actual = stats_current["ga"]

    # Calcula el peso asignado a la información de la temporada actual. Este peso aumenta conforme el equipo acumula más partidos.
    peso_actual = n / (n + k)

    # Calcula el peso asignado a la información de la temporada anterior. Este peso disminuye relativamente conforme aumenta n.
    peso_previo = k / (n + k)

    # Combina el promedio actual y el promedio previo para obtener el promedio ajustado de goles a favor.
    gf_ajustado = (
        peso_actual * gf_actual
        + peso_previo * gf_previo
    )

    # Combina el promedio actual y el promedio previo para obtener el promedio ajustado de goles en contra.
    ga_ajustado = (
        peso_actual * ga_actual
        + peso_previo * ga_previo
    )

    # Devuelve los promedios ajustados de goles a favor y en contra.
    return {
        "gf_avg": float(gf_ajustado),
        "ga_avg": float(ga_ajustado)
    }
