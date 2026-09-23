"""Capa visual del dashboard: paleta, tema de Plotly y una función por gráfica.

Reglas de diseño:
- El color codifica entidades fijas en todo el tablero: azul = modelo, naranja = mercado,
  verde = local, violeta = visitante, gris = contexto/empate. Paleta validada para
  daltonismo (separación entre pares y contraste sobre fondo blanco).
- El título de cada tarjeta dice el hallazgo; la gráfica no repite el título.
- Barras siempre desde cero; puntos y líneas cuando el cero no es relevante.
- Etiquetas directas selectivas; cuando hace falta leyenda, va en HTML en el subtítulo de
  la tarjeta (index.qmd), no dentro de Plotly: Plotly mide mal sus leyendas si la página
  del tablero está oculta cuando se dibuja la gráfica, y las entradas se enciman.
- Gráficas de lectura: sin zoom ni arrastre (un arrastre accidental dejaba la gráfica
  ampliada sin forma visible de regresar); los tooltips siguen activos.
- Las figuras se construyen dentro de funciones: el motor de Quarto imprime cualquier
  expresión suelta que devuelva una figura (p. ej. una línea `fig.update_layout(...)`).
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paleta ────────────────────────────────────────────────────────────────────
MODELO = "#2a78d6"
MERCADO = "#eb6834"
LOCAL = "#008300"
VISITA = "#4a3aa7"
CONTEXTO = "#898781"      # gris: empate, referencias y especificaciones secundarias
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TENUE = "#898781"
REJILLA = "#e1e0d9"
BASE = "#c3c2b7"
BANDA = "#f3f2ee"
AZULES = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

FUENTE = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False, "doubleClick": False}

PLANTILLA = go.layout.Template(layout=dict(
    font=dict(family=FUENTE, size=13, color=TINTA_2),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=[MODELO, MERCADO, LOCAL, VISITA, CONTEXTO],
    margin=dict(l=8, r=24, t=16, b=8),
    xaxis=dict(showgrid=True, gridcolor=REJILLA, gridwidth=1, zeroline=False, showline=False,
               ticks="", tickfont=dict(color=TENUE, size=12), title=dict(font=dict(size=12, color=TINTA_2)),
               automargin=True),
    yaxis=dict(showgrid=True, gridcolor=REJILLA, gridwidth=1, zeroline=False, showline=False,
               ticks="", tickfont=dict(color=TENUE, size=12), title=dict(font=dict(size=12, color=TINTA_2)),
               automargin=True),
    legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
                bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=TINTA_2), title=dict(text="")),
    hoverlabel=dict(bgcolor="white", bordercolor=REJILLA, font=dict(family=FUENTE, size=12, color=TINTA)),
    hovermode="closest",
    separators=".,",
))


def mostrar(fig: go.Figure) -> None:
    """Muestra una figura de sólo lectura, adaptada al tamaño de la tarjeta.

    Sin barra de herramientas, sin zoom ni arrastre (ejes fijos) y sin leyenda interna.
    """
    fig.update_layout(template=PLANTILLA, autosize=True, dragmode=False, showlegend=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.show(config=CONFIG)


def _pct(x, dec=1):
    return f"{x * 100:.{dec}f}%"


def _anotacion(fig, texto, x, y, ax=0, ay=-40, xref="x", yref="y", align="left", **kw):
    fig.add_annotation(text=texto, x=x, y=y, xref=xref, yref=yref, ax=ax, ay=ay,
                       showarrow=ax != 0 or ay != 0, arrowhead=0, arrowwidth=1, arrowcolor=TENUE,
                       font=dict(size=12, color=TINTA_2), align=align, bgcolor="rgba(255,255,255,0.85)", **kw)


# ── Resumen ───────────────────────────────────────────────────────────────────
def mejora_sobre_ingenua(metricas: pd.DataFrame) -> go.Figure:
    """Barras agrupadas: mejora de LogLoss respecto a la referencia ingenua."""
    filas = [("Mercado_apertura", "Mercado (cuotas de apertura)", MERCADO),
             ("M0_Base", "Modelo base M0 · 3 variables", MODELO),
             ("M4_Completo", "Modelo completo M4 · 9 variables", CONTEXTO)]
    fig = go.Figure()
    for conjunto, opacidad, etiqueta_periodo in [("Validación", 0.4, "Validación 2024/25"),
                                                  ("Prueba", 1.0, "Prueba 2025/26 – sep 2026")]:
        m = metricas[metricas["conjunto"] == conjunto].set_index("predictor")
        base = m.loc["Ingenua", "logloss"]
        x = [base - m.loc[p, "logloss"] for p, _, _ in filas]
        fig.add_bar(
            y=[n for _, n, _ in filas], x=x, orientation="h", name=etiqueta_periodo, showlegend=False,
            marker=dict(color=[c for _, _, c in filas], opacity=opacidad, cornerradius=4),
            text=[f"{v:.3f}" for v in x], textposition="outside", cliponaxis=False, constraintext="none",
            textfont=dict(color=TINTA_2, size=12),
            customdata=np.column_stack([[m.loc[p, "logloss"] for p, _, _ in filas],
                                        [m.loc[p, "aciertos"] for p, _, _ in filas]]),
            hovertemplate=("<b>%{y}</b><br>" + etiqueta_periodo +
                           "<br>Mejora vs. ingenua: %{x:.3f}<br>LogLoss: %{customdata[0]:.3f}"
                           "<br>Aciertos: %{customdata[1]:.1%}<extra></extra>"),
        )
    # El color identifica la entidad y la opacidad el periodo; la leyenda de la opacidad va en
    # el subtítulo de la tarjeta (HTML).
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.12, margin=dict(l=8, r=48, t=8, b=8))
    fig.update_xaxes(title_text="Reducción del LogLoss respecto a la referencia ingenua (más es mejor)",
                     rangemode="tozero", tickformat=".2f")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA, size=13))
    return fig


# ── ¿Qué ocurre? ──────────────────────────────────────────────────────────────
def resultados_por_temporada(t: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    # Bandas de los periodos de modelación: conectan el contexto con el modelo.
    # Las bandas de una sola temporada son angostas: su etiqueta va en vertical.
    for x0, x1, texto, angulo in [(2018.5, 2023.5, "Entrenamiento", 0), (2023.5, 2024.5, "Validación", -90),
                                  (2024.5, 2025.5, "Prueba", -90)]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=BANDA if texto != "Validación" else "#ebeae4",
                      line_width=0, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2, y=0.66 if angulo == 0 else 0.62, text=texto, showarrow=False,
                           yref="y", textangle=angulo, font=dict(size=11, color=TENUE))
    series = [("pct_local", "Local", LOCAL, 2.5), ("pct_visita", "Visitante", VISITA, 2.5),
              ("pct_empate", "Empate", CONTEXTO, 1.5)]
    for col, nombre, color, grosor in series:
        fig.add_scatter(
            x=t["temporada"], y=t[col], mode="lines+markers", name=nombre,
            line=dict(color=color, width=grosor), marker=dict(size=6, color=color, line=dict(color="white", width=1.5)),
            customdata=t["etiqueta"], hovertemplate="%{customdata}<br>" + nombre + ": %{y:.1%}<extra></extra>",
        )
        ultimo = t.iloc[-1]
        fig.add_annotation(x=ultimo["temporada"], y=ultimo[col], text=f"<b>{nombre}</b> {_pct(ultimo[col])}",
                           xanchor="left", xshift=8, showarrow=False, font=dict(size=12, color=TINTA_2))
    covid = t[t["temporada"] == 2020].iloc[0]
    # La anotación va en la franja vacía inferior para no tapar ninguna línea.
    fig.add_annotation(x=2020, y=covid["pct_local"], ax=2014.2, ay=0.1, axref="x", ayref="y",
                       text="<b>2020/21, estadios vacíos (COVID-19)</b><br>única temporada en que los "
                            "visitantes<br>ganan más partidos que los locales",
                       showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=TENUE, align="left",
                       font=dict(size=12, color=TINTA_2), bgcolor="rgba(255,255,255,0.9)")
    ticks = list(range(2001, 2026, 3))
    fig.update_layout(showlegend=False, hovermode="x unified", margin=dict(l=8, r=110, t=12, b=8))
    fig.update_xaxes(tickvals=ticks, ticktext=[f"{a}/{str(a + 1)[-2:]}" for a in ticks], showgrid=False,
                     range=[2000.5, 2025.8])
    fig.update_yaxes(range=[0, 0.68], tickformat=".0%", title_text="% de partidos de la temporada")
    return fig


def resultado_favorito(f: dict) -> go.Figure:
    etiquetas = ["Gana el favorito", "Empate", "Gana el no favorito"]
    valores = [f["gana_favorito"], f["empate"], f["gana_no_favorito"]]
    fig = go.Figure(go.Bar(
        y=etiquetas, x=valores, orientation="h", marker=dict(color=CONTEXTO, cornerradius=4),
        text=[_pct(v) for v in valores], textposition="outside", cliponaxis=False, constraintext="none",
        textfont=dict(color=TINTA_2, size=13), width=0.55,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(margin=dict(l=8, r=48, t=8, b=8))
    fig.update_xaxes(range=[0, 0.7], tickformat=".0%", title_text=f"% de {f['partidos']:,} partidos con cuotas de Bet365")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA, size=13))
    return fig


# ── Patrones ──────────────────────────────────────────────────────────────────
def resultado_por_diferencia_elo(t: pd.DataFrame, ventaja: float) -> go.Figure:
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=BASE, width=1))
    fig.add_annotation(x=0, y=0.01, text="Elo igual", showarrow=False, font=dict(size=11, color=TENUE), yref="y",
                       xanchor="left", yanchor="bottom", xshift=4)
    for col, nombre, color, grosor in [("pct_local", "Local", LOCAL, 2.5),
                                       ("pct_visita", "Visitante", VISITA, 2.5),
                                       ("pct_empate", "Empate", CONTEXTO, 1.5)]:
        fig.add_scatter(
            x=t["elo_mediana"], y=t[col], mode="lines+markers", name=nombre,
            line=dict(color=color, width=grosor, shape="spline", smoothing=0.4),
            marker=dict(size=8, color=color, line=dict(color="white", width=2)),
            customdata=np.column_stack([t["elo_min"], t["elo_max"], t["partidos"]]),
            hovertemplate=("Gana " + nombre.lower() + ": %{y:.1%}" if nombre != "Empate" else "Empate: %{y:.1%}")
                          + "<br>Diferencia Elo entre %{customdata[0]:.0f} y %{customdata[1]:.0f}"
                            "<br>%{customdata[2]} partidos<extra></extra>",
        )
        ultimo = t.iloc[-1]
        fig.add_annotation(x=ultimo["elo_mediana"], y=ultimo[col], text=f"<b>{nombre}</b>", xanchor="left",
                           xshift=10, showarrow=False, font=dict(size=12, color=TINTA_2))
    y_cruce = float(np.interp(-ventaja, t["elo_mediana"], t["pct_local"]))
    fig.add_scatter(x=[-ventaja], y=[y_cruce], mode="markers", hoverinfo="skip", showlegend=False,
                    marker=dict(size=9, color="white", line=dict(color=TINTA_2, width=2)))
    # Texto en la esquina superior izquierda, la única zona sin líneas.
    fig.add_annotation(x=-ventaja, y=y_cruce, ax=float(t["elo_mediana"].min()) - 20, ay=0.84, axref="x",
                       ayref="y", align="left",
                       text=f"Local y visitante tienen la misma probabilidad<br>de ganar cuando el local es "
                            f"≈{round(ventaja, -1):.0f} puntos<br>más débil: eso <b>vale jugar en casa</b>",
                       showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=TENUE,
                       font=dict(size=12, color=TINTA_2), bgcolor="rgba(255,255,255,0.9)", xanchor="left")
    fig.update_layout(showlegend=False, margin=dict(l=8, r=90, t=12, b=8))
    x0, x1 = float(t["elo_mediana"].min()), float(t["elo_mediana"].max())
    fig.update_xaxes(range=[x0 - 30, x1 + 30], title_text="Diferencia de Elo previa (local − visitante), por decil")
    fig.update_yaxes(range=[0, 0.9], tickformat=".0%", title_text="% de partidos")
    return fig


def calibracion_mercado(t: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=BASE, width=1), hoverinfo="skip",
                    name="Calibración perfecta")
    fig.add_scatter(
        x=t["prob_predicha"], y=t["frecuencia"], mode="markers", name="Bet365",
        marker=dict(size=10, color=MERCADO, line=dict(color="white", width=2)),
        customdata=t["n"],
        hovertemplate="Las cuotas decían %{x:.1%}<br>Ocurrió %{y:.1%}<br>%{customdata:,} casos<extra></extra>",
    )
    fig.add_annotation(x=0.97, y=0.06, text="diagonal = calibración perfecta", showarrow=False,
                       xanchor="right", font=dict(size=11, color=TENUE))
    alto = t.iloc[-1]
    fig.add_annotation(x=alto["prob_predicha"], y=alto["frecuencia"], ax=0.08, ay=0.82, axref="x", ayref="y",
                       text="Los grandes favoritos ganan<br>un poco más de lo que<br>dicen las cuotas",
                       xanchor="left", align="left", showarrow=True, arrowhead=0, arrowwidth=1,
                       arrowcolor=TENUE, font=dict(size=12, color=TINTA_2), bgcolor="rgba(255,255,255,0.9)")
    fig.update_layout(showlegend=False, margin=dict(l=8, r=16, t=12, b=8))
    # Ejes cuadrados para que la diagonal quede a 45°; `constrain="domain"` encoge el área de
    # dibujo en lugar de extender los rangos (evita ejes de −20 % a 120 %).
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="Probabilidad implícita en la cuota",
                     constrain="domain")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="Frecuencia observada", scaleanchor="x",
                     scaleratio=1, constrain="domain")
    return fig


def ajuste_poisson(p: pd.DataFrame) -> go.Figure:
    lados = ["Local", "Visitante"]
    titulos = []
    for lado in lados:
        s = p[p["lado"] == lado].iloc[0]
        titulos.append(f"<b>Goles del {lado.lower()}</b> · media {s['media']:.2f}, varianza {s['varianza']:.2f}")
    fig = make_subplots(rows=1, cols=2, subplot_titles=titulos, horizontal_spacing=0.08, shared_yaxes=True)
    for i, lado in enumerate(lados, start=1):
        s = p[p["lado"] == lado]
        fig.add_bar(x=s["etiqueta"], y=s["observado"], name="Observado", marker=dict(color=CONTEXTO, cornerradius=4),
                    showlegend=i == 1, hovertemplate="%{x} goles: %{y:.1%} observado<extra></extra>",
                    row=1, col=i, width=0.6)
        fig.add_scatter(x=s["etiqueta"], y=s["poisson"], name="Poisson con la misma media", mode="markers",
                        marker=dict(size=11, color=MODELO, symbol="diamond", line=dict(color="white", width=2)),
                        showlegend=i == 1, hovertemplate="%{x} goles: %{y:.1%} según Poisson<extra></extra>",
                        row=1, col=i)
    fig.update_layout(margin=dict(l=8, r=8, t=36, b=8), bargap=0.3)
    fig.update_annotations(font=dict(size=12, color=TINTA_2))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero", gridcolor=REJILLA)
    fig.update_xaxes(showgrid=False, title_text="Goles en el partido")
    return fig


# ── El modelo ─────────────────────────────────────────────────────────────────
def delta_vs_m0(d: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=TINTA_2, width=1))
    fig.add_annotation(x=0, y=1.04, yref="paper", text="<b>M0 (base)</b>", showarrow=False,
                       font=dict(size=12, color=TINTA_2))
    for r in d.itertuples():
        color = MERCADO if r.predictor == "Mercado_apertura" else CONTEXTO
        fig.add_scatter(x=[r.validacion, r.prueba], y=[r.nombre, r.nombre], mode="lines",
                        line=dict(color=color, width=2), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[r.validacion], y=[r.nombre], mode="markers", showlegend=False,
                        marker=dict(size=12, color="white", line=dict(color=color, width=2.5)),
                        hovertemplate=f"<b>{r.nombre}</b><br>Validación: %{{x:.4f}} vs M0<extra></extra>")
        fig.add_scatter(x=[r.prueba], y=[r.nombre], mode="markers", showlegend=False,
                        marker=dict(size=12, color=color, line=dict(color="white", width=2)),
                        hovertemplate=f"<b>{r.nombre}</b><br>Prueba: %{{x:.4f}} vs M0<extra></extra>")
    # La codificación (hueco = validación, relleno = prueba) se explica en el subtítulo (HTML).
    valores = np.concatenate([d["validacion"], d["prueba"]])
    x0, x1 = float(valores.min()) - 0.003, float(valores.max()) + 0.003
    fig.add_annotation(x=x0, y=-0.16, yref="paper", xanchor="left", text="← mejor que M0", showarrow=False,
                       font=dict(size=11, color=TENUE))
    fig.add_annotation(x=x1, y=-0.16, yref="paper", xanchor="right", text="peor que M0 →", showarrow=False,
                       font=dict(size=11, color=TENUE))
    fig.update_layout(margin=dict(l=8, r=16, t=28, b=40))
    fig.update_xaxes(title_text="", tickformat=".3f", zeroline=False, range=[x0, x1])
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA, size=13))
    return fig


def brecha_por_resultado(t: pd.DataFrame) -> go.Figure:
    colores = [MERCADO if v > 0 else MODELO for v in t["aporte"]]
    fig = go.Figure(go.Bar(
        y=t["resultado"], x=t["aporte"], orientation="h", marker=dict(color=colores, cornerradius=4), width=0.55,
        text=[f"{v:+.3f}" for v in t["aporte"]], textposition="outside", cliponaxis=False, constraintext="none",
        textfont=dict(color=TINTA_2, size=13),
        customdata=np.column_stack([t["partidos"], t["p_modelo"], t["p_mercado"]]),
        hovertemplate=("<b>%{y}</b> (%{customdata[0]} partidos)<br>Aporte a la brecha: %{x:.4f}"
                       "<br>Probabilidad media que le dio el modelo: %{customdata[1]:.1%}"
                       "<br>… y el mercado: %{customdata[2]:.1%}<extra></extra>"),
    ))
    fig.add_vline(x=0, line=dict(color=TINTA_2, width=1))
    limite = float(np.abs(t["aporte"]).max()) * 1.6
    fig.add_annotation(x=limite, y=-0.13, yref="paper", xanchor="right", yanchor="top",
                       text="el mercado fue mejor →", showarrow=False, font=dict(size=11, color=TENUE))
    fig.add_annotation(x=-limite, y=-0.13, yref="paper", xanchor="left", yanchor="top",
                       text="← el modelo fue mejor", showarrow=False, font=dict(size=11, color=TENUE))
    fig.update_layout(margin=dict(l=8, r=40, t=8, b=56))
    fig.update_xaxes(range=[-limite, limite], tickformat=".3f", zeroline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA, size=13))
    return fig


def efectos(e: pd.DataFrame) -> go.Figure:
    ecuaciones = list(dict.fromkeys(e["ecuacion"]))
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"<b>{x}</b>" for x in ecuaciones], horizontal_spacing=0.2)
    for i, ec in enumerate(ecuaciones, start=1):
        s = e[e["ecuacion"] == ec]
        fig.add_scatter(
            x=s["efecto"], y=s["etiqueta"], mode="markers", row=1, col=i, showlegend=False,
            marker=dict(size=11, color=MODELO, line=dict(color="white", width=2)),
            error_x=dict(type="data", symmetric=False, array=s["ic_sup"] - s["efecto"],
                         arrayminus=s["efecto"] - s["ic_inf"], color=MODELO, thickness=1.5, width=0),
            customdata=np.column_stack([s["ic_inf"], s["ic_sup"], s["p_valor"]]),
            hovertemplate=("%{y}<br>+1 desviación estándar → %{x:+.1%} goles esperados"
                           "<br>IC 95%: %{customdata[0]:+.1%} a %{customdata[1]:+.1%}"
                           "<br>p-valor: %{customdata[2]:.3f}<extra></extra>"),
        )
        fig.add_vline(x=0, line=dict(color=BASE, width=1), row=1, col=i)
    fig.update_layout(margin=dict(l=8, r=16, t=40, b=8))
    fig.update_annotations(font=dict(size=12, color=TINTA_2))
    fig.update_xaxes(tickformat="+.0%", title_text="Cambio en goles esperados (+1 desv. estándar)")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=TINTA, size=12))
    return fig


def calibracion_modelos(t: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=BASE, width=1), hoverinfo="skip",
                    showlegend=False)
    for pred, nombre, color in [("Mercado_apertura", "Mercado (apertura)", MERCADO), ("M0_Base", "Modelo M0", MODELO)]:
        s = t[t["predictor"] == pred]
        fig.add_scatter(x=s["prob_predicha"], y=s["frecuencia"], mode="lines+markers", name=nombre,
                        line=dict(color=color, width=2), marker=dict(size=9, color=color, line=dict(color="white", width=2)),
                        customdata=s["n"],
                        hovertemplate=nombre + "<br>Dijo %{x:.1%} → ocurrió %{y:.1%}<br>%{customdata} casos<extra></extra>")
    fig.update_layout(margin=dict(l=8, r=16, t=12, b=8))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="Probabilidad asignada")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="Frecuencia observada")
    return fig


def modelo_vs_mercado(d: pd.DataFrame) -> go.Figure:
    r = np.corrcoef(d["p_local_modelo"], d["p_local_mercado"])[0, 1]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=BASE, width=1), hoverinfo="skip",
                    showlegend=False)
    texto = [f"{html.escape(h)} – {html.escape(a)}<br>{f:%d-%m-%Y} · resultado: {res}"
             for h, a, f, res in zip(d["HomeTeam"], d["AwayTeam"], d["Date"], d["resultado"])]
    fig.add_scatter(x=d["p_local_mercado"], y=d["p_local_modelo"], mode="markers", showlegend=False,
                    marker=dict(size=8, color=TINTA_2, opacity=0.55, line=dict(color="white", width=1)),
                    text=texto, hovertemplate="%{text}<br>Mercado: %{x:.1%} · Modelo: %{y:.1%}<extra></extra>")
    fig.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", showarrow=False, align="left",
                       text=f"r = {r:.2f}", font=dict(size=14, color=TINTA))
    fig.update_layout(margin=dict(l=8, r=16, t=12, b=8))
    fig.update_xaxes(range=[0, 1], tickformat=".0%", title_text="P(victoria local) según el mercado")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="P(victoria local) según el modelo M0")
    return fig


# ── Tablas ────────────────────────────────────────────────────────────────────
def tabla_html(df: pd.DataFrame, clase: str = "tabla") -> str:
    """Tabla HTML sencilla; los textos se escapan porque vienen de archivos de datos."""
    cab = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    cuerpo = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in fila) + "</tr>"
        for fila in df.itertuples(index=False)
    )
    return f'<div class="tabla-contenedor"><table class="{clase}"><thead><tr>{cab}</tr></thead><tbody>{cuerpo}</tbody></table></div>'
