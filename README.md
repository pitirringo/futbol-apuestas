# Fútbol y mercados de apuestas · Premier League

Proyecto final del **Módulo 8 (Comunicación de resultados)** del Diplomado de Introducción Analítica
a la Ciencia de Datos.

## Equipo

- Castillo Rodríguez Daniel Arturo
- Castillo Santiago Erika Isabel
- Garduño Gutiérrez César Emiliano
- Gómez Mendoza Maximiliano
- Martínez Vega Eduardo
- Zacateco Tello María Fernanda

Datos: [Football-Data.co.uk](https://football-data.co.uk/englandm.php).

**Pregunta de investigación:** ¿Qué tan bien anticipan el resultado de un partido de la Premier League
(victoria local, empate o victoria visitante) las estadísticas disponibles antes del encuentro,
comparadas con las probabilidades implícitas en las cuotas de apuestas?

**Respuesta corta:** un modelo de Poisson con variables previas al partido (diferencia de Elo y goles
a favor y en contra de la temporada) acierta casi la mitad de los partidos de prueba y recupera la mayor
parte de la mejora que consiguen las cuotas sobre una referencia ingenua, pero no supera al mercado.
Las cifras exactas están en el tablero, que las recalcula desde el código cada vez que se publica.

- **Dashboard:** https://pitirringo.github.io/futbol-apuestas/
- **Repositorio:** https://github.com/pitirringo/futbol-apuestas
- **Reporte:** Se anexa en el repositorio bajo el nombre "Reporte.pdf"

## Estructura

```text
Codigo/
├── proyecto_mod_8/
│   ├── Limpieza de datos.ipynb   consolidación de los CSV de Football-Data
│   ├── E0_consolidado.csv        base consolidada: un partido por fila, de 2001/02 a la fecha
│   ├── wc_predictor.py           Elo, forma reciente y goles de la temporada (K y k calibrados)
│   ├── premier_training_data.csv variables previas al partido (2019/20 en adelante)
│   └── Analisis.ipynb            calibración de K y k, modelos de Poisson M0–M4, validación, prueba y mercado
└── Documentacion/                reporte técnico del modelo (LaTeX y PDF)
Dashboard-o-pagina/               tablero (Quarto + Python)
.github/workflows/                publicación automática del tablero en GitHub Pages
```

### Cómo se conectan las partes

El trabajo avanza en cadena: cada etapa toma lo que produjo la anterior.

```text
Datos crudos (Football-Data.co.uk, 26 archivos E0.csv)
      │  Codigo/proyecto_mod_8/Limpieza de datos.ipynb
      ▼
E0_consolidado.csv          9,540 partidos, del 18/08/2001 al 14/09/2026
      │  Codigo/proyecto_mod_8/Analisis.ipynb (usa wc_predictor.py)
      ▼
premier_training_data.csv   2,696 partidos, del 09/08/2019 al 14/09/2026
+ modelos M0–M4, validación, prueba y comparación con el mercado
      │  Dashboard-o-pagina/ (Quarto + datos_dashboard.py + graficas.py)
      ▼
Dashboard-o-pagina/_site/index.html   el tablero que se publica en GitHub Pages
```

Aparte, `Codigo/Documentacion/main.tex` genera `Modulo_8.pdf`, el reporte técnico.

## Reproducibilidad

Esta sección explica cómo obtener los mismos resultados que presentamos en el proyecto: la base de
datos, los modelos, las cifras, el tablero y el reporte.

Para asegurarnos de que las instrucciones funcionan, repetimos todo el proceso desde cero el 30 de
septiembre de 2026, en Linux, con Python 3.10.20 y Quarto 1.8.25, a partir del commit `16c9e4b`. El
notebook de análisis volvió a producir todas las cifras que habíamos guardado, y el archivo
`premier_training_data.csv` se generó idéntico al original.

> **Importante:** `E0_consolidado.csv` y `premier_training_data.csv` ya están incluidos en el
> repositorio. Para reproducir nuestros resultados no necesitas descargar nada más; con los pasos 2,
> 3 y 4 es suficiente. El paso 5, que reconstruye la base desde los datos crudos, es opcional.

### 1. Qué necesitas instalar

| Programa | Versión que usamos | Para qué sirve |
|---|---|---|
| Git | cualquiera reciente | descargar el repositorio |
| [Python](https://www.python.org/downloads/release/python-31011/) | 3.10.x | todo el análisis |
| [Quarto](https://github.com/quarto-dev/quarto-cli/releases/tag/v1.8.25) | 1.8.25 | construir el tablero |

Las versiones de los paquetes de Python están fijas en `Dashboard-o-pagina/requirements.txt`:

```text
pandas==2.2.3        numpy==2.2.3         scipy==1.15.2
statsmodels==0.14.4  scikit-learn==1.6.1  plotly==5.24.1
jupyter, pyyaml
```

### 2. Preparar el entorno

**2.1 Descargar el repositorio**

```bash
git clone https://github.com/pitirringo/futbol-apuestas.git
cd futbol-apuestas
```

**2.2 Confirmar que los datos no han cambiado**

Cada archivo tiene una "huella" (SHA-256) que cambia en cuanto se modifica un solo carácter. Puedes
calcularla así:

```bash
# Linux
sha256sum Codigo/proyecto_mod_8/E0_consolidado.csv Codigo/proyecto_mod_8/premier_training_data.csv

# macOS
shasum -a 256 Codigo/proyecto_mod_8/E0_consolidado.csv Codigo/proyecto_mod_8/premier_training_data.csv
```

```powershell
# Windows (PowerShell)
Get-FileHash Codigo\proyecto_mod_8\E0_consolidado.csv -Algorithm SHA256
Get-FileHash Codigo\proyecto_mod_8\premier_training_data.csv -Algorithm SHA256
```

Deberías obtener:

| Archivo | SHA-256 |
|---|---|
| `E0_consolidado.csv` | `7cea84b123c6924f453463fe3e6936631a87ecc469e2eed649f0c1874bb48e33` |
| `premier_training_data.csv` | `37de4c4cab439e7448f567eb722476885a7770ab53ffcc561b6a70ffbb5c3aa8` |

Si no coinciden, el archivo se modificó en algún momento. Lo más común es haberlo abierto y guardado
en Excel, que cambia el formato de las fechas y los decimales. En ese caso, vuelve a descargarlo del
repositorio y evita guardar los CSV desde Excel.

**2.3 Crear un entorno virtual con Python 3.10**

Un entorno virtual mantiene los paquetes del proyecto separados del resto de tu computadora, para que
otras versiones no interfieran.

```bash
# Linux o macOS
python3.10 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows (PowerShell)
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# Si prefieres conda
conda create -n futbol python=3.10 -y
conda activate futbol
```

**2.4 Instalar los paquetes**

```bash
python -m pip install --upgrade pip
pip install -r Dashboard-o-pagina/requirements.txt
pip install nbconvert matplotlib
```

**2.5 Instalar Quarto y comprobar que funciona**

Después de instalar Quarto 1.8.25, ejecuta:

```bash
quarto --version        # debe mostrar 1.8.25
quarto check jupyter    # debe reconocer el Python del entorno
```

### 3. Reproducir el análisis (`Analisis.ipynb`)

El notebook tiene que ejecutarse desde su propia carpeta, porque busca `E0_consolidado.csv` y
`wc_predictor.py` en ese mismo lugar.

La forma más segura es hacerlo desde la terminal, porque así todas las celdas corren en orden, de
principio a fin:

```bash
cd Codigo/proyecto_mod_8
jupyter nbconvert --to notebook --execute Analisis.ipynb \
        --output Analisis_ejecutado.ipynb \
        --ExecutePreprocessor.timeout=3600
```

Si prefieres Jupyter o VS Code, abre `Codigo/proyecto_mod_8/Analisis.ipynb`, elige el kernel del
entorno `.venv` y usa **Restart Kernel and Run All Cells**. Conviene no ejecutar celdas sueltas ni en
desorden, porque algunas dependen de variables creadas en celdas anteriores.

El análisis no tiene ningún componente aleatorio: los datos se dividen por fecha y los modelos de
Poisson se estiman con un método determinista. Por eso no hace falta fijar una semilla, y cada
ejecución da exactamente los mismos números.

**3.1 Cifras que deberías ver**

```text
Histórico: 9540 partidos, 2001-08-18 a 2026-09-14
K Elo seleccionado: 15 | k shrinkage seleccionado: 0
Log-Loss temporal promedio (calibración): 0.959558
Base construida: 2700 partidos, 2696 tras limpieza
Entrenamiento: 1897 partidos, 2019-08-09 a 2024-05-19
Validación:     380 partidos, 2024-08-16 a 2025-05-25
Prueba:         419 partidos, 2025-08-15 a 2026-09-14

Modelo de referencia simple: LogLoss_1X2 = 1.0793610651
```

| Log-Loss 1X2 | Mercado (apertura) | M0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|---|
| Validación (2024/25) | 0.970552 | 0.989547 | 0.987603 | 0.980718 | 0.982319 | 0.978612 |
| Prueba (2025/26 en adelante) | 1.020000 | 1.033076 | 1.034699 | 1.037331 | 1.033868 | 1.036739 |

Predicción de ejemplo (Arsenal vs Man City, corte 2026-09-20, modelo M0):
`P_home 0.43646 | P_draw 0.25000 | P_away 0.31354 | marcador 1-1`

Hay dos diferencias normales que no deben preocuparte: la fecha y hora que statsmodels imprime en sus
resúmenes (`Date:` y `Time:`), y cambios mínimos a partir del decimal 14, que se deben a la forma en
que la computadora redondea. Cualquier otra diferencia suele indicar que una versión de algún paquete
no coincide o que los datos se modificaron.

**3.2 (Opcional) Comprobar que `premier_training_data.csv` se genera igual**

El notebook construye una tabla llamada `training_data`, y el tablero usa esa misma tabla guardada como
`premier_training_data.csv`. Para comprobar que son idénticas, agrega esta celda al final del notebook
y ejecútalo completo:

```python
import pandas as pd
ref = pd.read_csv("premier_training_data.csv", parse_dates=["Date"])
nuevo = training_data[ref.columns].reset_index(drop=True)
pd.testing.assert_frame_equal(nuevo, ref, check_exact=False,
                              rtol=1e-9, check_dtype=False)
print("OK", ref.shape)
```

Si todo está bien, verás el mensaje `OK (2696, 24)`. Si quieres volver a generar el archivo, basta con
quitar el comentario del bloque `training_data.to_csv(...)` que ya está en el notebook.

**3.3 (Opcional) Repetir la búsqueda completa de hiperparámetros**

Para que el notebook corra rápido, sólo evalúa los valores que terminamos eligiendo: K_Elo = 15 y
k_shrinkage = 0. Si quieres repetir la búsqueda completa, cambia estas dos líneas en la celda de
configuración:

```python
K_ELO_CANDIDATOS = [15, 20, 25, 30, 35, 40, 45]   # en lugar de [15]
K_SHRINKAGE_CANDIDATOS = [0, 5, 10, 15, 20]       # en lugar de [0]
```

Eso da 35 combinaciones y puede tardar varias decenas de minutos. Al final, el notebook debería volver
a elegir K = 15 y k = 0.

### 4. Reproducir el tablero (`Dashboard-o-pagina`)

Desde la carpeta principal del repositorio, y con el entorno activado:

```bash
# Linux o macOS
export QUARTO_PYTHON="$(which python)"
quarto render Dashboard-o-pagina
```

```powershell
# Windows (PowerShell)
$env:QUARTO_PYTHON = (Get-Command python).Source
quarto render Dashboard-o-pagina
```

El tablero queda en `Dashboard-o-pagina/_site/index.html` y se abre con cualquier navegador.

Algunos detalles útiles:

- La variable `QUARTO_PYTHON` le indica a Quarto qué Python usar. Si no se define, Quarto puede tomar
  otro Python instalado en la computadora, y ésa es la causa más común de errores como
  `module not found`.
- El tablero no copia cifras a mano: `datos_dashboard.py` las vuelve a calcular a partir de los
  archivos de `Codigo/proyecto_mod_8`. Si mueves esa carpeta, indica su nueva ubicación con la
  variable `RUTA_CODIGO`, por ejemplo `export RUTA_CODIGO=/ruta/a/proyecto_mod_8`.
- La carpeta `_site/` se genera cada vez y no se sube a GitHub (está en `.gitignore`).

**4.1 Publicación automática en GitHub Pages**

Cada vez que alguien sube cambios a la rama `main`, GitHub repite este mismo paso en un servidor limpio
(Ubuntu, Python 3.10 y Quarto 1.8.25) siguiendo el archivo `.github/workflows/publicar-dashboard.yml`,
y publica el resultado en https://pitirringo.github.io/futbol-apuestas/.

Si trabajas en una copia propia del repositorio (un fork), activa la publicación en
**Settings › Pages › Source: GitHub Actions**. También puedes lanzarla a mano desde la pestaña
**Actions** con **Run workflow**.

Este proceso funciona, además, como una prueba de reproducibilidad: si el tablero no se puede
construir en una máquina nueva, no se publica.

### 5. (Opcional) Reconstruir `E0_consolidado.csv` desde los datos crudos

**5.1 Descargar los 26 archivos de Football-Data.co.uk**

Los datos vienen de https://football-data.co.uk/englandm.php. Cada temporada se descarga en una
dirección de esta forma:

```text
https://www.football-data.co.uk/mmz4281/AABB/E0.csv
```

donde `AABB` son los dos últimos dígitos de los dos años de la temporada (por ejemplo, `2425` para
2024/25).

Todos los archivos se llaman `E0.csv`, así que el navegador los renombra al descargarlos: `E0.csv`,
`E0 (1).csv`, `E0 (2).csv`, etcétera. El notebook espera justamente esos nombres, descargados de la
temporada más reciente a la más antigua:

| Nombre local | Temporada | Código `AABB` |
|---|---|---|
| `E0.csv` | 2026/27 | 2627 |
| `E0 (1).csv` | 2025/26 | 2526 |
| `E0 (2).csv` | 2024/25 | 2425 |
| `E0 (3).csv` | 2023/24 | 2324 |
| `E0 (4).csv` | 2022/23 | 2223 |
| `E0 (5).csv` | 2021/22 | 2122 |
| `E0 (6).csv` | 2020/21 | 2021 |
| `E0 (7).csv` | 2019/20 | 1920 |
| `E0 (8).csv` | 2018/19 | 1819 |
| `E0 (9).csv` | 2017/18 | 1718 |
| `E0 (10).csv` | 2016/17 | 1617 |
| `E0 (11).csv` | 2015/16 | 1516 |
| `E0 (12).csv` | 2014/15 | 1415 |
| `E0 (13).csv` | 2013/14 | 1314 |
| `E0 (14).csv` | 2012/13 | 1213 |
| `E0 (15).csv` | 2011/12 | 1112 |
| `E0 (16).csv` | 2010/11 | 1011 |
| `E0 (17).csv` | 2009/10 | 0910 |
| `E0 (18).csv` | 2008/09 | 0809 |
| `E0 (19).csv` | 2007/08 | 0708 |
| `E0 (20).csv` | 2006/07 | 0607 |
| `E0 (21).csv` | 2005/06 | 0506 |
| `E0 (22).csv` | 2004/05 | 0405 |
| `E0 (23).csv` | 2003/04 | 0304 |
| `E0 (24).csv` | 2002/03 | 0203 |
| `E0 (25).csv` | 2001/02 | 0102 |

Confirmamos este orden revisando `E0_consolidado.csv`: sus filas van de la temporada 2026/27 hacia
atrás, hasta 2001/02.

Guarda los 26 archivos juntos en una misma carpeta; por ejemplo, `datos_crudos/` dentro del
repositorio.

**5.2 Ajustar y ejecutar `Limpieza de datos.ipynb`**

Primero, en la primera celda de código, cambia la ruta que apunta a la carpeta de descargas de uno de
nosotros:

```python
carpeta = Path("C:/Users/roski/Downloads")
```

por la carpeta donde guardaste los archivos, por ejemplo:

```python
carpeta = Path("../../datos_crudos")
```

Después ejecuta el notebook completo desde `Codigo/proyecto_mod_8`:

```bash
jupyter nbconvert --to notebook --execute "Limpieza de datos.ipynb" \
        --output Limpieza_ejecutado.ipynb
```

El notebook guarda el resultado como `E0_consolidado_final.csv`, pero el resto del proyecto busca
`E0_consolidado.csv`. Puedes compararlos o simplemente renombrar el nuevo:

```bash
mv E0_consolidado_final.csv E0_consolidado.csv
```

Antes de seguir, revisa que el resultado tenga 9,540 filas y 34 columnas. Si falta alguno de los 26
archivos, el notebook lo salta sin avisar y la base queda incompleta.

**5.3 Por qué este paso no garantiza exactamente la misma base**

Hay tres razones:

- La temporada 2026/27 sigue en curso y Football-Data actualiza su archivo cada semana. Nuestra versión
  llega hasta el 14/09/2026 (40 partidos de esa temporada). Para obtener la misma base, hay que
  quedarse sólo con los partidos hasta esa fecha, por ejemplo con esta línea después de unir los
  archivos:

  ```python
  df_consolidado = df_consolidado[
      pd.to_datetime(df_consolidado["Date"], format="mixed", dayfirst=True) <= "2026-09-14"]
  ```

- De vez en cuando, Football-Data corrige datos de temporadas pasadas (cuotas, árbitros o xG), así que
  una descarga futura podría diferir en algún valor.
- El periodo de prueba del análisis empieza el 01/08/2025 pero no tiene fecha de cierre. Si se agregan
  partidos nuevos, cambian las cifras de prueba, la comparación con el mercado y el tablero.

Por eso, para reproducir los resultados que reportamos, la referencia es el `E0_consolidado.csv`
incluido en el repositorio, cuya huella puedes comprobar en el paso 2.2. Este paso 5 sirve para
revisar cómo se construyó esa base, no para reemplazarla.

## Uso de herramientas de IA

En este proyecto nos apoyamos en herramientas de inteligencia artificial para tareas como revisar
código, redactar documentación y verificar la reproducibilidad. Todo el contenido fue revisado por el
equipo, que es responsable del análisis, los resultados y las conclusiones.


