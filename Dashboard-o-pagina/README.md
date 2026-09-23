# Dashboard · Premier League: estadística vs. mercado de apuestas

Visualización final del **Proyecto 1 (Fútbol y mercados de apuestas)**, Módulo 8 del Diplomado de
Introducción Analítica a la Ciencia de Datos.

- **URL publicada:** https://pitirringo.github.io/futbol-apuestas/
- **Repositorio:** https://github.com/pitirringo/futbol-apuestas
- **Herramienta:** Quarto (formato `dashboard`) + Python (pandas, statsmodels, Plotly) + Observable JS
- **Publicación:** GitHub Actions → GitHub Pages, en cada `git push` a `main`

El tablero tiene seis páginas:

| Página | Contenido |
|---|---|
| Resumen | Pregunta, respuesta corta, indicadores principales y conclusión |
| ¿Qué ocurre? | Resultados históricos, ventaja de local y qué tan seguido gana el favorito |
| Patrones | Diferencia de Elo contra resultado, calibración de las cuotas y distribución de goles |
| El modelo | Cómo funciona, comparación de especificaciones y comparación con el mercado |
| Explora un partido | Simulador interactivo con el modelo M0 |
| Datos y método | Datos, limpieza, variables, evaluación, limitaciones y reproducibilidad |

## Estructura

```text
06_proyecto/                        ← raíz del repositorio
├── .github/workflows/
│   └── publicar-dashboard.yml      ← construye y publica el tablero
├── Codigo/proyecto_mod_8/          ← trabajo del equipo de Código (no se modifica)
│   ├── wc_predictor.py
│   ├── E0_consolidado.csv
│   ├── premier_training_data.csv
│   └── Analisis.ipynb, Limpieza de datos.ipynb
└── Dashboard-o-pagina/
    ├── _quarto.yml                 ← proyecto Quarto (salida en _site/)
    ├── index.qmd                   ← estructura, textos y orden de la historia
    ├── datos_dashboard.py          ← todas las cifras (reutiliza el código del equipo)
    ├── graficas.py                 ← paleta, tema y una función por gráfica
    ├── estilos.scss                ← tema visual sobre `cosmo`
    ├── redibujar.html              ← redibuja Plotly al mostrar cada página
    ├── requirements.txt            ← versiones fijas de Python
    └── _site/                      ← salida generada (ignorada por git)
```

`datos_dashboard.py` importa `wc_predictor.py` desde `../Codigo/proyecto_mod_8` y reproduce las
funciones de `Analisis.ipynb`. Si el código vive en otra carpeta, basta con definir la variable de
entorno `RUTA_CODIGO`.


