import numpy as np
import pandas as pd

# ── Constantes del modelo ─────────────────────────────────────────────────────

N_SIMS       = 30_000
AVG_WC_GOALS = 1.3619  #(2.52 sobre 2, el primedio histórico de goles por partido)
ELO_INIT     = 1500
ELO_K        = 30

HIST_URL = 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv'

NAME_MAP = {
    'USA': 'United States',
    'United States of America': 'United States',
    'Korea Republic': 'South Korea',
    'Türkiye': 'Turkey',
    'Côte d’Ivoire': 'Ivory Coast',
    "Côte d'Ivoire": 'Ivory Coast',
    'Czechia': 'Czech Republic',
    'Curaçao': 'Curacao',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Bosnia and Herzegowina': 'Bosnia and Herzegovina',
    'Congo DR': 'DR Congo',
    'Democratic Republic of Congo': 'DR Congo',
}

INJURY_FACTOR = {}


# ── Carga de histórico ────────────────────────────────────────────────────────

def load_history(
    path="E0_consolidado.csv"
):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_score",
        "FTAG": "away_score"
    })

    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        format="mixed"
    )

    df = df.dropna(subset=[
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score"
    ])

    return df.sort_values("date").reset_index(drop=True)

def calcular_parametros_liga(df):
    """
    Calcula los promedios históricos de goles y
    los factores de localía a partir del DataFrame.
    """
# Calculamos los facotres de loclía como la media de los goles del local y el visitante sobre dos y ponderados por el promedio general de goles por equipo. Esto nos da una idea de cuánto más probable es que un equipo marque goles cuando juega en casa en comparación con cuando juega fuera.

    avg_home_goals = df["home_score"].mean()
    avg_away_goals = df["away_score"].mean()

    avg_team_goals = (
        avg_home_goals + avg_away_goals
    ) / 2

    home_factor = avg_home_goals / avg_team_goals
    away_factor = avg_away_goals / avg_team_goals

    return {
        "avg_team_goals": avg_team_goals,
        "home_factor": home_factor,
        "away_factor": away_factor
    }

# Cargar el histórico
df = load_history()

# Calcular parámetros de la liga
parametros = calcular_parametros_liga(df)

avg_team_goals = parametros["avg_team_goals"]

HOME_FACTOR = parametros["home_factor"]
AWAY_FACTOR = parametros["away_factor"]
# ── Elo ────────────────────────────────────────────────────────────────────────

def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def update_elo(r_a, r_b, score_a, k=ELO_K):
    exp_a = expected_score(r_a, r_b)
    return r_a + k * (score_a - exp_a)

def build_elo(df, elo_init=ELO_INIT):
    """
    Calcula el rating Elo de cada club de la Premier League
    utilizando los resultados históricos.

    Los partidos se procesan cronológicamente.
    Cada equipo comienza con un Elo inicial de 1500.
    """

    ratings = {}

    for _, row in df.sort_values("date").iterrows():

        h = row["home_team"]
        a = row["away_team"]

        hs = row["home_score"]
        as_ = row["away_score"]

        # Elo anterior al partido
        rh = ratings.get(h, elo_init)
        ra = ratings.get(a, elo_init)

        # Resultado del partido
        if hs > as_:
            sh, sa = 1, 0

        elif hs < as_:
            sh, sa = 0, 1

        else:
            sh, sa = 0.5, 0.5

        # Actualización de Elo
        ratings[h] = update_elo(rh, ra, sh)
        ratings[a] = update_elo(ra, rh, sa)

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
    tmp = df[
        (df["home_team"] == team) |
        (df["away_team"] == team)
    ].sort_values("date").tail(n)

    # Si no hay partidos disponibles
    if tmp.empty:
        return {
            "gf": avg_team_goals,
            "ga": avg_team_goals,
            "shots_for": np.nan,
            "shots_against": np.nan,
            "sot_for": np.nan,
            "sot_against": np.nan
        }

    # Inicializar listas
    gf = []
    ga = []

    shots_for = []
    shots_against = []

    sot_for = []
    sot_against = []

    # Recorrer los partidos
    for _, row in tmp.iterrows():

        if row["home_team"] == team:

            # Goles
            gf.append(row["home_score"])
            ga.append(row["away_score"])

            # Tiros
            shots_for.append(row["HS"])
            shots_against.append(row["AS"])

            # Tiros a puerta
            sot_for.append(row["HST"])
            sot_against.append(row["AST"])

        else:

            # Goles
            gf.append(row["away_score"])
            ga.append(row["home_score"])

            # Tiros
            shots_for.append(row["AS"])
            shots_against.append(row["HS"])

            # Tiros a puerta
            sot_for.append(row["AST"])
            sot_against.append(row["HST"])

    # Pesos geométricos
    w = np.array([
        decay ** (len(gf) - 1 - i)
        for i in range(len(gf))
    ])

    # Normalizar los pesos
    w = w / w.sum()

    # Promedios ponderados
    return {
        "gf": float(np.average(gf, weights=w)),
        "ga": float(np.average(ga, weights=w)),

        "shots_for": float(
            np.average(shots_for, weights=w)
        ),

        "shots_against": float(
            np.average(shots_against, weights=w)
        ),

        "sot_for": float(
            np.average(sot_for, weights=w)
        ),

        "sot_against": float(
            np.average(sot_against, weights=w)
        )
    }

def season_stats(df, team, as_of_date, k=10):
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

    fecha = pd.to_datetime(as_of_date)

    # Evitar utilizar partidos futuros
    df_pre = df[df["date"] < fecha].copy()

    # Identificar el inicio de la temporada actual
    if fecha.month >= 8:
        season_year = fecha.year
    else:
        season_year = fecha.year - 1

    season_start = pd.Timestamp(
        year=season_year,
        month=8,
        day=1
    )

    previous_start = pd.Timestamp(
        year=season_year - 1,
        month=8,
        day=1
    )

    # Partidos de la temporada actual

    current = df_pre[
        df_pre["date"] >= season_start
    ]

    current = current[
        (current["home_team"] == team) |
        (current["away_team"] == team)
    ]

    #Partidos de la temporada anterior

    previous_league = df_pre[
        (df_pre["date"] >= previous_start) &
        (df_pre["date"] < season_start)
    ]

    previous = previous_league[
        (previous_league["home_team"] == team) |
        (previous_league["away_team"] == team)
    ]

    # 3. Función auxiliar para calcular GF/GA

    def calculate_stats(matches):

        if matches.empty:
            return None

        gf = np.where(
            matches["home_team"] == team,
            matches["home_score"],
            matches["away_score"]
        )

        ga = np.where(
            matches["home_team"] == team,
            matches["away_score"],
            matches["home_score"]
        )

        return {
            "gf": float(np.mean(gf)),
            "ga": float(np.mean(ga))
        }

    # Estadísticas de ambas temporadas
    stats_current = calculate_stats(current)
    stats_previous = calculate_stats(previous)

    # Construir el promedio previo


    if stats_previous is None:

        # Si el equipo no jugó la temporada anterior,
        # utilizar el promedio de goles de esa liga.

        if not previous_league.empty:

            league_avg = (
                previous_league["home_score"].sum()
                + previous_league["away_score"].sum()
            ) / (2 * len(previous_league))

        else:

            # Respaldo para temporadas sin datos previos
            historical = df_pre[
                df_pre["date"] < season_start
            ]

            if historical.empty:
                raise ValueError(
                    "No existe histórico anterior suficiente "
                    "para calcular el promedio previo."
                )

            league_avg = (
                historical["home_score"].sum()
                + historical["away_score"].sum()
            ) / (2 * len(historical))

        gf_previo = league_avg
        ga_previo = league_avg

    else:

        gf_previo = stats_previous["gf"]
        ga_previo = stats_previous["ga"]


    # Aplicar shrinkage


    n = len(current)

    if n == 0:

        return {
            "gf_avg": gf_previo,
            "ga_avg": ga_previo
        }

    gf_actual = stats_current["gf"]
    ga_actual = stats_current["ga"]

    peso_actual = n / (n + k)
    peso_previo = k / (n + k)

    gf_ajustado = (
        peso_actual * gf_actual
        + peso_previo * gf_previo
    )

    ga_ajustado = (
        peso_actual * ga_actual
        + peso_previo * ga_previo
    )

    return {
        "gf_avg": float(gf_ajustado),
        "ga_avg": float(ga_ajustado)
    }


# ── λ (goles esperados) ────────────────────────────────────────────────────────

def get_lambda(attacker_stats, defender_stats, attacker_form, defender_form,
               elo_att, elo_def, home_factor=1.0, injury_factor=1.0,
               avg_goals=avg_team_goals):
    """
    λ combina tres fuentes:
      50% stats del torneo actual (attacker_stats / defender_stats — de API-Football
          si las tienes, o las mismas de recent_form si no)
      35% forma histórica ponderada (attacker_form / defender_form)
      15% ratio de Elo

    attacker_stats / defender_stats deben traer 'gf_avg' y 'ga_avg'.
    """
    gf_api = attacker_stats.get('gf_avg', avg_goals)
    ga_api = defender_stats.get('ga_avg', avg_goals)
    lam_api = (gf_api / avg_goals) * (ga_api / avg_goals) * avg_goals

    gf_hist = attacker_form['gf']
    ga_hist = defender_form['ga']
    lam_hist = (gf_hist / avg_goals) * (ga_hist / avg_goals) * avg_goals

    elo_ratio = 10 ** ((elo_att - elo_def) / 800)
    lam_elo = avg_goals * elo_ratio

    lam = 0.50 * lam_api + 0.35 * lam_hist + 0.15 * lam_elo
    lam *= home_factor
    lam *= injury_factor

    return round(max(lam, 0.2), 3)


# ── Simulación de un partido ───────────────────────────────────────────────────

def simular_partido(df, elo, home, away, ronda='', n_sims=N_SIMS,
                    stats_home=None, stats_away=None, as_of_date=None):
    """
    Corre Monte Carlo sobre Poisson(λ_home) vs Poisson(λ_away) y devuelve
    un dict con probabilidades a 90 min, prórroga/penales y marcador más
    probable.

    stats_home / stats_away son opcionales: si vienen de API-Football
    (gf_avg, ga_avg del torneo actual) se usan tal cual; si no, se calculan
    del histórico con team_base_stats (modo bracket_completo.ipynb).
    """
    # Fecha de predicción
    if as_of_date is None:
        fecha = df["date"].max() + pd.Timedelta(days=1)
    else:
        fecha = pd.to_datetime(as_of_date)

    # Histórico disponible antes de la predicción
    df_pre = df[df["date"] < fecha].copy()

    # Estadísticas de la temporada actual
    s_h = (
        stats_home
        if stats_home is not None
        else season_stats(df_pre, home, fecha)
    )

    s_a = (
        stats_away
        if stats_away is not None
        else season_stats(df_pre, away, fecha)
    )

    # Forma reciente de ambos equipos
    f_h = recent_form(df_pre, home)
    f_a = recent_form(df_pre, away)

    # Elo de ambos equipos
    if as_of_date is None:
        elo_actual = elo
    else:
        elo_actual = build_elo(df_pre)

    elo_h = elo_actual.get(home, ELO_INIT)
    elo_a = elo_actual.get(away, ELO_INIT)  


    # Goles esperados del equipo local
    lam_h = get_lambda(
        s_h, s_a, f_h, f_a, elo_h, elo_a,
        home_factor=HOME_FACTOR,
        injury_factor=INJURY_FACTOR.get(home, 1.0)
    )

    # Goles esperados del equipo visitante
    lam_a = get_lambda(
        s_a, s_h, f_a, f_h, elo_a, elo_h,
        home_factor=AWAY_FACTOR,
        injury_factor=INJURY_FACTOR.get(away, 1.0)
    )

    gh = np.random.poisson(lam_h, n_sims)
    ga = np.random.poisson(lam_a, n_sims)

    p_h = np.mean(gh > ga)
    p_d = np.mean(gh == ga)
    p_a = np.mean(gh < ga)

    total = gh + ga
    over25 = np.mean(total > 2.5)
    over35 = np.mean(total > 3.5)
    btts   = np.mean((gh > 0) & (ga > 0))

    # Prórroga y penales — no hay empate posible en rondas eliminatorias
    p_pen_h = 0.51 if elo_h >= elo_a else 0.49
    p_pen_a = 1 - p_pen_h
    extra = p_d * 0.50

    p_h_total = p_h + extra * p_pen_h + extra * (lam_h / (lam_h + lam_a))
    p_a_total = p_a + extra * p_pen_a + extra * (lam_a / (lam_h + lam_a))

    ganador  = home if p_h_total >= p_a_total else away
    perdedor = away if ganador == home else home

    marcador = (pd.Series([f'{h}-{a}' for h, a in zip(gh, ga)])
                  .value_counts(normalize=True).idxmax())

    return {
        'Ronda': ronda,
        'Local': home,
        'Visitante': away,
        'Ganador': ganador,
        'Perdedor': perdedor,
        'P local 90': p_h,
        'P empate 90': p_d,
        'P visita 90': p_a,
        'P pasa local': p_h_total,
        'P pasa visita': p_a_total,
        'Prob ganador': max(p_h_total, p_a_total),
        'Marcador probable': marcador,
        'Over 2.5': over25,
        'Over 3.5': over35,
        'BTTS': btts,
        'xG local': lam_h,
        'xG visita': lam_a,
        'Elo local': elo_h,
        'Elo visita': elo_a,
    }

from wc_predictor import load_history, build_elo

# Cargar histórico
df = load_history()

# Construir Elo
elo = build_elo(df)

# Mostrar ratings ordenados
elo_ordenado = sorted(
    elo.items(),
    key=lambda x: x[1],
    reverse=True
)

for equipo, rating in elo_ordenado:
    print(f"{equipo:25s} {rating:.2f}")