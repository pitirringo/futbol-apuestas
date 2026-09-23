# Fútbol y mercados de apuestas · Premier League

Proyecto final del **Módulo 8 (Comunicación de resultados)** del Diplomado de Introducción Analítica
a la Ciencia de Datos.

**Pregunta de investigación:** ¿Qué tan bien anticipan el resultado de un partido de la Premier League
(victoria local, empate o victoria visitante) las estadísticas disponibles antes del encuentro,
comparadas con las probabilidades implícitas en las cuotas de apuestas?

**Respuesta corta:** un modelo de Poisson con tres variables previas al partido (diferencia de Elo y
goles a favor y en contra ajustados) acierta el 48 % de los partidos de prueba y logra el 84 % de la
mejora que consiguen las cuotas sobre una referencia ingenua, pero no supera al mercado.

- **Dashboard:** https://pitirringo.github.io/futbol-apuestas/
- **Repositorio:** https://github.com/pitirringo/futbol-apuestas
- **Reporte:** _lo entrega el equipo de Reporte_

## Estructura

```text
Codigo/
├── proyecto_mod_8/
│   ├── Limpieza de datos.ipynb   consolidación de los CSV de Football-Data
│   ├── E0_consolidado.csv        9,450 partidos (2001/02 a sep-2026), 33 variables
│   ├── wc_predictor.py           Elo, forma reciente y promedios con shrinkage
│   ├── premier_training_data.csv variables previas al partido (caché, 2019/20 en adelante)
│   └── Analisis.ipynb            modelos de Poisson M0–M4, validación, prueba y comparación con el mercado
└── Documentacion/                reporte técnico del modelo (LaTeX y PDF)
Dashboard-o-pagina/               tablero (Quarto + Python)
.github/workflows/                publicación automática del tablero en GitHub Pages
```

## Reproducibilidad

Requisitos: Python 3.10 y los paquetes de `Dashboard-o-pagina/requirements.txt`
(pandas, numpy, scipy, statsmodels, scikit-learn, plotly, jupyter).

```bash
pip install -r Dashboard-o-pagina/requirements.txt
```

**Datos.** Los archivos originales se descargan de
[Football-Data.co.uk](https://football-data.co.uk/englandm.php) (archivos `E0`, temporadas 2001/02 a
2026/27). `Codigo/proyecto_mod_8/Limpieza de datos.ipynb` los consolida en `E0_consolidado.csv`, que ya
está incluido en el repositorio.

**Análisis.** Ejecutar `Codigo/proyecto_mod_8/Analisis.ipynb` desde su propia carpeta. Usa la caché
`premier_training_data.csv`

**Dashboard.** Requiere además [Quarto](https://quarto.org) 1.4 o superior:

```bash
quarto render Dashboard-o-pagina
```



## Equipo

- Castillo Rodríguez Daniel Arturo
- Castillo Santiago Erika Isabel
- Garduño Gutiérrez César Emiliano
- Gómez Mendoza Maximiliano
- Martínez Vega Eduardo
- Zacateco Tello María Fernanda

Datos: Football-Data.co.uk. 
