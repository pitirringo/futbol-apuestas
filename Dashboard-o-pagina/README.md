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

## Reproducir localmente

Requisitos: Python 3.10 y Quarto ≥ 1.4 (el que trae RStudio sirve).

```bash
pip install -r Dashboard-o-pagina/requirements.txt
quarto render Dashboard-o-pagina
```

En la computadora de César, con las rutas del `CLAUDE.md`:

```powershell
& "C:\Program Files\RStudio\resources\app\bin\quarto\bin\quarto.exe" render "Dashboard-o-pagina"
```

El resultado queda en `Dashboard-o-pagina/_site/index.html`. Para verlo con el simulador
funcionando, sírvelo con un servidor local (los navegadores bloquean algunos recursos al abrir el
archivo directamente):

```bash
python -m http.server 8000 --directory Dashboard-o-pagina/_site
```

y abre `http://localhost:8000`. El render tarda unos 2–3 minutos.

**Comprobación automática:** la pestaña *Datos y método → Reproducibilidad* compara las cifras del
tablero con las del notebook y del reporte técnico. Si el equipo de Código cambia el modelo o los
datos, esa tabla mostrará ✗ hasta que se actualicen los valores de referencia
(`REPORTADO_NOTEBOOK` en `datos_dashboard.py`).

## Publicar en GitHub Pages

Mismo procedimiento que el ejemplo de clase (`04_githubactions`):

1. Subir la carpeta `06_proyecto` como repositorio (debe incluir `Codigo/` con los CSV).
2. En GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Hacer `git push` a `main` (o ejecutar el flujo a mano en **Actions**).
4. Al terminar los jobs `build` y `deploy`, la URL aparece en el job `deploy`
   (normalmente `https://USUARIO.github.io/REPOSITORIO/`).
5. Opcional: **About → ⚙ → Use your GitHub Pages website** para dejar la URL visible.

El botón de GitHub en la barra del tablero apunta al repositorio; se define con `nav-buttons` en el
YAML de `index.qmd`.

## Problemas conocidos (y cómo se resolvieron)

| Síntoma | Causa | Solución aplicada |
|---|---|---|
| `No module named 'yaml'` al renderizar | Quarto necesita PyYAML para ejecutar Python | `pyyaml` en `requirements.txt` |
| Una gráfica aparece dos veces | El motor de Quarto imprime expresiones sueltas (p. ej. `fig.update_layout(...)`) | Las figuras se construyen dentro de funciones en `graficas.py` |
| El simulador dice "sim is not defined" | `ojs_define()` no funciona en celdas con `include: false` | La celda con `ojs_define` usa `echo: false` |
| Leyendas encimadas o etiquetas ausentes | Plotly mide textos cuando la página aún no tiene tamaño | `redibujar.html` redibuja al cargar y al cambiar de página |
| Diagrama Mermaid vacío | Mermaid no dibuja en páginas ocultas al cargar | Diagrama hecho con HTML y CSS |
| Barra de desplazamiento dentro de las tarjetas | El subtítulo y la gráfica se repartían la altura | Regla CSS: el texto que acompaña a una celda no crece |
| Página en blanco al abrir un enlace a una sección | Ids con acentos (`qué-ocurre`) | Ids ASCII explícitos: `#que-ocurre`, `#datos`, … |
