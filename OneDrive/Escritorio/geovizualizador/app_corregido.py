import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import io, base64
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GeoVisualizador Turístico de Aysén",
    page_icon="🗺️",
    layout="wide"
)

st.markdown("""
<div style="
background:linear-gradient(90deg,#1B5E20,#43A047);
padding:25px;
border-radius:15px;
text-align:center;
color:white;
margin-bottom:20px;">

<h1 style="margin:0;font-size:42px;">
🗺️ GeoVisualizador Turístico de la Región de Aysén
</h1>

<p style="font-size:20px;margin-top:10px;">
Explora áreas protegidas, atractivos turísticos y el relieve de la región mediante un visor SIG interactivo.
</p>

</div>
""", unsafe_allow_html=True)

DATA = Path("data")
CRS_METRICO = None  # se calcula dinámicamente por capa con estimate_utm_crs()

import unicodedata

def quitar_tildes(texto):
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

# ─────────────────────────────────────────────────────────────
# PASO 1: Paletas cartográficas
# ─────────────────────────────────────────────────────────────

PALETA_ASP_FIJA = {
    "parque":             "#1A237E",   # índigo/azul marino — contrasta bien con el DEM verde
    "reserva":            "#F4511E",   # naranjo profundo
    "monumento":          "#F9A825",   # ámbar/dorado
    "santuario":          "#00BCD4",   # cian brillante
    "biosfera":           "#8E24AA",   # púrpura
    "ramsar":             "#1E88E5",   # azul (coherente con humedales)
    "prioritario":        "#D81B60",   # magenta
    "bien nacional":      "#FFEB3B",   # amarillo — Bien Nacional Protegido
}
PALETA_ASP_RESPALDO = [
    "#5C6BC0", "#FF7043", "#8D6E63", "#EC407A", "#26C6DA",
    "#FFCA28", "#7E57C2", "#EF5350", "#42A5F5", "#AB47BC",
]

ESTILOS_TIPO_TRANSPORTE = {
    "autopista":   {"color": "#CC0000", "weight": 6},
    "ruta":        {"color": "#E05000", "weight": 4},
    "camino":      {"color": "#FF8800", "weight": 3},
    "pavimentado": {"color": "#E05000", "weight": 4},
    "ripio":       {"color": "#DAA520", "weight": 2},
    "tierra":      {"color": "#8B6914", "weight": 2},
    "sendero":     {"color": "#A0522D", "weight": 1},
    "ferrocarril": {"color": "#333333", "weight": 3},
    "default":     {"color": "#888888", "weight": 2},
}

ESTILOS_TIPO_HIDRO = {
    "rio":      {"color": "#1565C0", "weight": 3.5},
    "estero":   {"color": "#1E88E5", "weight": 2.0},
    "quebrada": {"color": "#64B5F6", "weight": 1.5},
    "canal":    {"color": "#00ACC1", "weight": 1.5},
    "lago":     {"color": "#0D47A1", "weight": 1.5},
    "laguna":   {"color": "#1976D2", "weight": 1.5},
    "humedal":  {"color": "#00897B", "weight": 1.5},
    "default":  {"color": "#2196F3", "weight": 1.5},
}

PALETA_ATRACTIVOS_FIJA = {
    "sitios naturales":            "#2E7D32",
    "museos manifestaciones":      "#6A1B9A",
    "folklore":                    "#EF6C00",
    "realizaciones tecnicas":      "#0277BD",
    "rutas y circuitos":           "#AD1457",
    "acontecimientos programados": "#F9A825",
}
PALETA_ATRACTIVOS_RESPALDO = ["#455A64", "#00838F", "#795548", "#C62828"]

# Los nombres oficiales SERNATUR son muy largos para una leyenda de mapa
# (ej. "Realizaciones Técnicas y Científicas Contemporáneas y Culturales
# Históricas"), así que se muestran acortados sin perder el sentido.
ABREVIACIONES_ETIQUETAS = {
    "sitios naturales":            "Sitios naturales",
    "museos manifestaciones":      "Museos y sitios históricos",
    "folklore":                    "Folklore",
    "realizaciones tecnicas":      "Realizaciones técnicas/científicas",
    "rutas y circuitos":           "Rutas y circuitos turísticos",
    "acontecimientos programados": "Acontecimientos programados",
}


def formatear_etiqueta(etiqueta):
    if etiqueta == "None":
        return "Sin categoría (dato faltante)"
    et_low = quitar_tildes(str(etiqueta).lower())
    for key, bonito in ABREVIACIONES_ETIQUETAS.items():
        if key in et_low:
            return bonito
    # Si no hay abreviación conocida, se muestra en formato Título en vez de MAYÚSCULAS
    return str(etiqueta).strip().title()

COLORMAP_DEM = [
    (0.00, "#08306B"),
    (0.05, "#1B5E20"),
    (0.20, "#4CAF50"),
    (0.40, "#AED581"),
    (0.60, "#DAA520"),
    (0.75, "#8B4513"),
    (0.90, "#D2B48C"),
    (1.00, "#FFFAFA"),
]

# ─────────────────────────────────────────────────────────────
# PASO 2: Funciones de color dinámico y estilos
# ─────────────────────────────────────────────────────────────

def construir_mapa_colores(serie, paleta_fija=None, paleta_respaldo=None, color_nulo="#9E9E9E"):
    paleta_fija = paleta_fija or {}
    paleta_respaldo = paleta_respaldo or ["#888888"]
    valores = sorted(serie.dropna().unique().tolist())
    color_map = {}
    i = 0
    for v in valores:
        v_low = quitar_tildes(str(v).lower())
        color_asignado = None
        for key, color in paleta_fija.items():
            if key in v_low:
                color_asignado = color
                break
        if color_asignado is None:
            color_asignado = paleta_respaldo[i % len(paleta_respaldo)]
            i += 1
        color_map[str(v).strip()] = color_asignado

    return color_map


def crear_style_categorico(color_map, col, fill=True, weight=1.8):
    def style_fn(feature):
        val = str(feature["properties"].get(col, "")).strip()
        color = color_map.get(val, "#AAAAAA")
        if fill:
            return {"fillColor": color, "color": "#ffffff", "weight": weight, "fillOpacity": 0.8}
        else:
            return {"color": color, "weight": 3, "opacity": 0.9}
    return style_fn


def crear_style_por_diccionario(col, estilos_dict):
    def style_fn(feature):
        val = quitar_tildes(str(feature["properties"].get(col, "")).lower())
        for key, vals in estilos_dict.items():
            if key in val:
                return {"color": vals["color"], "weight": vals["weight"], "opacity": 0.9}
        d = estilos_dict["default"]
        return {"color": d["color"], "weight": d["weight"], "opacity": 0.9}
    return style_fn


# ─────────────────────────────────────────────────────────────
# PASO 3: Leyendas HTML (versión "sección" — sin posicionamiento propio)
# ─────────────────────────────────────────────────────────────
# En vez de que cada leyenda flote sola con position:fixed (lo que hacía que
# se encimaran y se cortaran fuera del mapa), cada función ahora devuelve
# sólo el bloque de contenido. Todas las secciones se juntan al final en
# UN único panel fijo, con scroll, en la esquina superior derecha.

def _simbolo_pin(color, size=14):
    """Pin/marcador tipo 'gota' — para puntos como Atractivos Turísticos."""
    return f"""<svg width="{size}" height="{size + 5}" viewBox="0 0 24 32"
         style="flex-shrink:0;margin-right:7px;margin-top:1px;">
      <path d="M12 0C5.4 0 0 5.4 0 12c0 9 12 20 12 20s12-11 12-20c0-6.6-5.4-12-12-12z"
            fill="{color}" stroke="#ffffff" stroke-width="1.3"/>
      <circle cx="12" cy="12" r="4.3" fill="#ffffff"/>
    </svg>"""


def _simbolo_area(color, size=15):
    """Polígono irregular — representa una zona/área (ideal para SNASPE / áreas protegidas)."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24"
         style="flex-shrink:0;margin-right:7px;margin-top:1px;">
      <polygon points="4,9 10,3 18,5 21,13 15,21 6,19 2,14"
               fill="{color}" stroke="#ffffff" stroke-width="1.3" stroke-linejoin="round"/>
    </svg>"""


def _simbolo_linea(color, size=15):
    """Trazo — representa una capa de líneas (ríos, caminos)."""
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24"
         style="flex-shrink:0;margin-right:7px;margin-top:6px;">
      <path d="M2 18 L9 8 L15 14 L22 4" fill="none"
            stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>"""


def _simbolo_cuadrado(color, size=14):
    return f"""<div style="background:{color};width:{size}px;height:{size}px;
                border:1px solid rgba(255,255,255,0.6);margin-right:7px;margin-top:2px;
                border-radius:2px;flex-shrink:0;"></div>"""


def seccion_leyenda_categorica(titulo, color_map, icono="🔲", forma="cuadrado"):
    """forma: 'cuadrado' (default), 'pin' (marcador de punto), 'area' (polígono de zona),
    o 'linea' (trazo, para capas lineales)."""
    constructores = {
        "cuadrado": _simbolo_cuadrado,
        "pin": _simbolo_pin,
        "area": _simbolo_area,
        "linea": _simbolo_linea,
    }
    construir_simbolo = constructores.get(forma, _simbolo_cuadrado)

    items = ""
    for etiqueta, color in sorted(color_map.items()):
        simbolo = construir_simbolo(color)
        etiqueta_mostrar = formatear_etiqueta(etiqueta)
        items += f"""
        <div style="display:flex;align-items:flex-start;margin:4px 0;">
          {simbolo}
          <span style="font-size:11px;color:#e8e8e8;line-height:1.35;
                       flex:1;min-width:0;overflow-wrap:break-word;
                       white-space:normal;">{etiqueta_mostrar}</span>
        </div>"""
    return f"""
    <div style="margin-bottom:12px;">
      <div style="font-size:12px;font-weight:600;color:#ffffff;
                  display:flex;align-items:center;gap:6px;margin-bottom:6px;">
        <span>{icono}</span><span>{titulo}</span>
      </div>
      <div style="height:1px;background:rgba(255,255,255,0.18);margin-bottom:7px;"></div>
      {items}
    </div>"""


def seccion_leyenda_dem(dem_min, dem_max):
    stops = ", ".join([f"{color} {int(pct*100)}%" for pct, color in COLORMAP_DEM])
    gradient = f"linear-gradient(to top, {stops})"
    return f"""
    <div style="margin-bottom:12px;">
      <div style="font-size:12px;font-weight:600;color:#ffffff;
                  display:flex;align-items:center;gap:6px;margin-bottom:6px;">
        <span>🏔️</span><span>Elevación (m)</span>
      </div>
      <div style="height:1px;background:rgba(255,255,255,0.18);margin-bottom:7px;"></div>
      <div style="display:flex;align-items:stretch;gap:8px;">
        <div style="width:20px;height:130px;background:{gradient};
                    border:1px solid rgba(255,255,255,0.35);border-radius:3px;flex-shrink:0;"></div>
        <div style="display:flex;flex-direction:column;
                    justify-content:space-between;font-size:11px;color:#e8e8e8;">
          <span><b>{int(dem_max)} m</b></span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.75)} m</span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.50)} m</span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.25)} m</span>
          <span><b>{int(dem_min)} m</b></span>
        </div>
      </div>
    </div>"""


def panel_leyendas_html(secciones):
    """Junta todas las secciones en un único panel. Colapsar/expandir usa
    <details>/<summary> nativo del navegador (sin JavaScript, 100% confiable
    incluso dentro de iframes de componentes de Streamlit). Reposicionar usa
    radio buttons ocultos + CSS puro (":checked ~"), también sin JavaScript."""
    if not secciones:
        return ""
    contenido = "".join(secciones)
    return f"""
    <input type="radio" name="panel-pos" id="pos-tr" checked style="display:none;">
    <input type="radio" name="panel-pos" id="pos-tl" style="display:none;">
    <input type="radio" name="panel-pos" id="pos-br" style="display:none;">
    <input type="radio" name="panel-pos" id="pos-bl" style="display:none;">

    <div style="
        position: fixed;
        top: 6px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1000;
        display: flex;
        gap: 3px;
        background: rgba(24,26,30,0.85);
        padding: 4px;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.15);
        font-family: Arial, sans-serif;">
      <label for="pos-tl" title="Mover arriba-izquierda" style="cursor:pointer;width:22px;height:22px;
             display:flex;align-items:center;justify-content:center;color:#fff;font-size:13px;
             border-radius:4px;">↖</label>
      <label for="pos-tr" title="Mover arriba-derecha" style="cursor:pointer;width:22px;height:22px;
             display:flex;align-items:center;justify-content:center;color:#fff;font-size:13px;
             border-radius:4px;">↗</label>
      <label for="pos-bl" title="Mover abajo-izquierda" style="cursor:pointer;width:22px;height:22px;
             display:flex;align-items:center;justify-content:center;color:#fff;font-size:13px;
             border-radius:4px;">↙</label>
      <label for="pos-br" title="Mover abajo-derecha" style="cursor:pointer;width:22px;height:22px;
             display:flex;align-items:center;justify-content:center;color:#fff;font-size:13px;
             border-radius:4px;">↘</label>
    </div>

    <details id="panel-leyenda" open style="
        position: fixed;
        top: 75px;
        right: 10px;
        z-index: 999;
        background: rgba(24,26,30,0.94);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 4px 16px rgba(0,0,0,0.45);
        width: 240px;
        box-sizing: border-box;
        font-family: 'Segoe UI', Arial, sans-serif;
        color: #e8e8e8;
        overflow: hidden;">
      <summary style="
          list-style: none;
          padding: 10px 16px;
          background: linear-gradient(90deg,#1B5E20,#2E7D32);
          font-size: 13px;
          font-weight: 700;
          color: #ffffff;
          letter-spacing: 0.3px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: space-between;
          user-select: none;">
        <span>🗺️ Leyenda</span>
        <span style="font-size:14px;">▾</span>
      </summary>
      <div style="padding: 12px 16px; max-height: 55vh; overflow-y: auto;">
        {contenido}
      </div>
    </details>

    <style>
      #panel-leyenda summary::-webkit-details-marker {{ display: none; }}
      #panel-leyenda summary::marker {{ content: ""; }}
      #pos-tl:checked ~ #panel-leyenda {{ top: 75px; left: 10px; right: auto; bottom: auto; }}
      #pos-tr:checked ~ #panel-leyenda {{ top: 75px; right: 10px; left: auto; bottom: auto; }}
      #pos-bl:checked ~ #panel-leyenda {{ bottom: 10px; left: 10px; top: auto; right: auto; }}
      #pos-br:checked ~ #panel-leyenda {{ bottom: 10px; right: 10px; top: auto; left: auto; }}
    </style>
    """

# PASO 4: Raster → ImageOverlay con colormap DEM
# ─────────────────────────────────────────────────────────────

def aplicar_colormap_dem(band, nodata):
    posiciones = [p for p, _ in COLORMAP_DEM]
    colores = [c for _, c in COLORMAP_DEM]
    cmap = mcolors.LinearSegmentedColormap.from_list("dem", list(zip(posiciones, colores)))

    mascara = (band == nodata) if nodata is not None else np.zeros_like(band, dtype=bool)
    valid = band[~mascara]
    dem_min = float(valid.min()) if len(valid) > 0 else 0
    dem_max = float(valid.max()) if len(valid) > 0 else 1

    norm = mcolors.Normalize(vmin=dem_min, vmax=dem_max)
    rgba = cmap(norm(band))
    rgba[mascara, 3] = 0
    rgba[~mascara, 3] = 0.82
    img_array = (rgba * 255).astype(np.uint8)
    return img_array, dem_min, dem_max


@st.cache_data(show_spinner="Procesando raster...")
def raster_a_overlay(raster_path, es_dem=False):
    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            data = np.zeros((src.count, height, width), dtype=np.float32)
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=data[i - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear,
                )
            bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
        else:
            data = src.read().astype(np.float32)
            bounds_wgs84 = src.bounds

        nodata = src.nodata
        dem_min = dem_max = None

        if es_dem:
            img_array, dem_min, dem_max = aplicar_colormap_dem(data[0], nodata)
        else:
            if src.count >= 3:
                rgb = data[:3].copy()
            else:
                rgb = np.stack([data[0]] * 3)
            for i in range(3):
                band = rgb[i]
                mask = (band == nodata) if nodata is not None else np.zeros_like(band, dtype=bool)
                valid = band[~mask]
                if len(valid) > 0:
                    mn, mx = np.percentile(valid, 2), np.percentile(valid, 98)
                    rgb[i] = np.clip((band - mn) / (mx - mn + 1e-10), 0, 1)
                rgb[i][mask] = 0
            base = (np.transpose(rgb, (1, 2, 0)) * 255).astype(np.uint8)
            alpha = np.full((base.shape[0], base.shape[1]), 200, dtype=np.uint8)
            if nodata is not None:
                alpha[data[0] == nodata] = 0
            img_array = np.dstack([base, alpha])

        img_pil = Image.fromarray(img_array)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        return img_b64, bounds, dem_min, dem_max

# ─────────────────────────────────────────────────────────────
# Cargar archivos
# ─────────────────────────────────────────────────────────────

archivos_vec = list(DATA.glob("*.gpkg")) + list(DATA.glob("*.shp")) + list(DATA.glob("*.geojson"))


@st.cache_data(show_spinner="Cargando capas vectoriales...")
def cargar_capas_vectoriales(lista_archivos):
    capas_cargadas = {}
    for archivo in lista_archivos:
        nombre = archivo.stem.replace("_", " ")
        try:
            gdf = gpd.read_file(archivo)
            if gdf.crs is not None:
                gdf = gdf.to_crs(4326)
            capas_cargadas[nombre] = gdf
        except Exception as e:
            st.warning(f"No fue posible cargar {archivo.name}: {e}")
    return capas_cargadas


capas = cargar_capas_vectoriales(archivos_vec)

archivos_raster = list(DATA.glob("*.tif")) + list(DATA.glob("*.tiff")) + list(DATA.glob("*.img"))
rasters = {archivo.stem.replace("_", " "): archivo for archivo in archivos_raster}

if not capas and not rasters:
    st.info(
        "📂 No se encontraron datos en la carpeta `data/`. Agrega tus capas "
        "(.gpkg/.shp/.geojson y .tif) con nombres que incluyan palabras clave como "
        "`protegidas`, `hidrografia`, `vialidad` y `dem` para que la app las reconozca "
        "y estilice automáticamente."
    )

def detectar_columna(gdf, candidatos):
    cols_low = {c.lower(): c for c in gdf.columns}
    for cand in candidatos:
        if cand.lower() in cols_low:
            return cols_low[cand.lower()]
    return None

def columnas_categoricas(gdf, max_categorias=25):
    out = []
    for c in gdf.columns:
        if c == gdf.geometry.name:
            continue
        if gdf[c].dtype == object or str(gdf[c].dtype).startswith("category"):
            n = gdf[c].nunique(dropna=True)
            if 1 < n <= max_categorias:
                out.append(c)
    return out

def columnas_numericas(gdf):
    return [c for c in gdf.columns if c != gdf.geometry.name and pd.api.types.is_numeric_dtype(gdf[c])]


ETIQUETAS_BONITAS_ASP = {
    "parque":        "Parque",
    "reserva":       "Reserva",
    "monumento":     "Monumento",
    "santuario":     "Santuario",
    "biosfera":      "Biosfera",
    "ramsar":        "Ramsar",
    "prioritario":   "Sitio Prioritario",
    "bien nacional": "Bien Nacional Protegido",
}


def inferir_categoria_desde_nombre(gdf, col_categoria, col_nombre, paleta_fija):
    """Cuando el campo de categoría (Tipo_Snasp/Categoria) está vacío, intenta
    rescatarlo buscando palabras clave (parque, reserva, monumento, etc.) en
    el nombre del área — muy común que el nombre oficial ya incluya la
    categoría (ej. 'Parque Nacional Perito Moreno')."""
    def _inferir(row):
        val = row.get(col_categoria) if col_categoria else None
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
        if col_nombre:
            nombre_val = row.get(col_nombre)
            if pd.notna(nombre_val):
                nombre_low = quitar_tildes(str(nombre_val).lower())
                for key in paleta_fija.keys():
                    if key in nombre_low:
                        return ETIQUETAS_BONITAS_ASP.get(key, key.title())
        return None

    gdf = gdf.copy()
    gdf["_categoria_snasp"] = gdf.apply(_inferir, axis=1)
    return gdf

# ─────────────────────────────────────────────────────────────
# Sidebar: capas
# ─────────────────────────────────────────────────────────────

st.sidebar.title("🗂️ Capas disponibles")
st.sidebar.subheader("🗺️ Vectores")
capas_activas = [n for n in capas if st.sidebar.checkbox(n, value=True, key=f"vec_{n}")]

st.sidebar.subheader("🛰️ Rasters")
rasters_activos = [n for n in rasters if st.sidebar.checkbox(n, value=True, key=f"rst_{n}")]

# ─────────────────────────────────────────────────────────────
# Sidebar: Panel de Análisis (Estadísticas + Filtro + Gráfico)
# ─────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Panel de Análisis")

capa_analisis = None
col_filtro = None
valores_filtro = None
col_grafico = None

capas_vectoriales_activas = [n for n in capas_activas if n in capas]

if capas_vectoriales_activas:
    capa_analisis = st.sidebar.selectbox(
        "Selecciona una capa para analizar", capas_vectoriales_activas, key="sel_capa_analisis"
    )
    gdf_sel = capas[capa_analisis]

    with st.sidebar.expander("📊 Estadísticas", expanded=True):
        geom_type = gdf_sel.geom_type.iloc[0] if len(gdf_sel) > 0 else ""
        st.write(f"**N° de elementos:** {len(gdf_sel)}")

        try:
            gdf_metrico = gdf_sel.to_crs(gdf_sel.estimate_utm_crs())
            if "Polygon" in geom_type:
                area_ha = gdf_metrico.geometry.area.sum() / 10_000
                st.write(f"**Área total:** {area_ha:,.1f} ha")
            elif "LineString" in geom_type:
                long_km = gdf_metrico.geometry.length.sum() / 1_000
                st.write(f"**Longitud total:** {long_km:,.1f} km")
        except Exception:
            pass

        num_cols = columnas_numericas(gdf_sel)
        if num_cols:
            col_num = st.selectbox("Atributo numérico", num_cols, key="col_num_stats")
            serie = gdf_sel[col_num].dropna()
            if len(serie) > 0:
                c1, c2 = st.columns(2)
                c1.metric("Mínimo", f"{serie.min():,.1f}")
                c2.metric("Máximo", f"{serie.max():,.1f}")
                st.write(f"**Promedio:** {serie.mean():,.1f}")

    cat_cols = columnas_categoricas(gdf_sel)
    if cat_cols:
        with st.sidebar.expander("🎚️ Filtro interactivo"):
            col_filtro = st.selectbox("Filtrar por atributo", ["(sin filtro)"] + cat_cols, key="col_filtro")
            if col_filtro != "(sin filtro)":
                valores_unicos = sorted(gdf_sel[col_filtro].dropna().unique().tolist())
                valores_filtro = st.multiselect(
                    "Valores a mostrar", valores_unicos, default=valores_unicos, key="valores_filtro"
                )
            else:
                col_filtro = None

    if cat_cols:
        with st.sidebar.expander("📈 Gráfico"):
            col_grafico = st.selectbox("Variable a graficar", cat_cols, key="col_grafico")
            conteo = gdf_sel[col_grafico].value_counts()
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.barh(conteo.index.astype(str), conteo.values, color="#2E7D32")
            ax.set_xlabel("N° de elementos")
            ax.set_title(f"{capa_analisis}\npor {col_grafico}", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
else:
    st.sidebar.caption("Activa al menos una capa vectorial para habilitar el análisis.")

# ─────────────────────────────────────────────────────────────
# Mapa base
# ─────────────────────────────────────────────────────────────

centro = [-46.5, -72.9]  # Región de Aysén (centro aproximado)
for gdf in capas.values():
    if len(gdf) > 0:
        try:
            c = gdf.union_all().centroid
        except AttributeError:
            c = gdf.unary_union.centroid
        centro = [c.y, c.x]
        break

m = folium.Map(location=centro, zoom_start=9, tiles="OpenStreetMap")
folium.TileLayer("CartoDB positron", name="Mapa claro").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Mapa oscuro").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satélite",
).add_to(m)

secciones_leyenda = []  # se acumulan aquí y se renderizan en UN solo panel al final

# ─────────────────────────────────────────────────────────────
# PASO 5: Agregar rasters
# ─────────────────────────────────────────────────────────────

for nombre in rasters_activos:
    nombre_low = nombre.lower()
    es_dem = any(k in nombre_low for k in ["dem", "dtm", "elevacion", "mde"])
    try:
        with st.spinner(f"Cargando raster: {nombre}..."):
            img_b64, bounds, dem_min, dem_max = raster_a_overlay(rasters[nombre], es_dem=es_dem)

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=bounds,
            opacity=0.75,
            name=f"🛰 {nombre}",
        ).add_to(m)

        if es_dem and dem_min is not None:
            secciones_leyenda.append(seccion_leyenda_dem(dem_min, dem_max))
    except Exception as e:
        st.warning(f"No fue posible cargar raster '{nombre}': {e}")

# ─────────────────────────────────────────────────────────────
# PASO 6: Agregar vectores con estilos, filtro y leyendas
# ─────────────────────────────────────────────────────────────

def _es_geom_puntual(nombre):
    gdf_tmp = capas[nombre]
    return len(gdf_tmp) > 0 and gdf_tmp.geom_type.iloc[0] == "Point"

# Los polígonos y líneas se dibujan primero; las capas de puntos (atractivos,
# sitios, etc.) se dibujan al final para que Leaflet las apile por encima.
orden_capas = sorted(capas_activas, key=_es_geom_puntual)

for nombre in orden_capas:
    gdf = capas[nombre]
    nombre_low = nombre.lower()

    if nombre == capa_analisis and col_filtro and valores_filtro is not None:
        gdf = gdf[gdf[col_filtro].isin(valores_filtro)]

    if len(gdf) == 0:
        continue

    geom_base = gdf.geom_type.iloc[0]
    col_snasp = detectar_columna(gdf, ["Tipo_Snasp", "Categoria", "CATEGORIA", "Tipo", "TIPO", "Nombre_Cat"])
    col_dren = detectar_columna(gdf, ["Dren_Tipo"])

    es_protegidas = any(k in nombre_low for k in ["protegid", "snasp", "snap", "asp"]) or (
        col_snasp is not None and "snasp" in col_snasp.lower()
    )
    es_hidro = (not es_protegidas) and (
        any(k in nombre_low for k in ["hidr", "rio", "agua"]) or col_dren is not None
    )
    es_vial = (not es_protegidas and not es_hidro) and any(
        k in nombre_low for k in ["vial", "camino", "acceso", "transport", "ruta"]
    )
    es_limite = (
        not (es_protegidas or es_hidro or es_vial)
        and "Polygon" in geom_base
        and len(gdf) <= 5
        and detectar_columna(gdf, ["codregion", "cir_sena", "Region"]) is not None
    )
    es_atractivo = (
        not (es_protegidas or es_hidro or es_vial or es_limite)
        and geom_base == "Point"
        and detectar_columna(gdf, ["CATEGORIA", "Jerarquia"]) is not None
    )

    # ── Áreas Protegidas (SNASPE/SNAP/ASP) ─────────────────────
    if es_protegidas:
        col_nombre_asp = detectar_columna(gdf, ["Nombre", "NOMBRE", "nombre"])

        if col_snasp or col_nombre_asp:
            gdf = inferir_categoria_desde_nombre(gdf, col_snasp, col_nombre_asp, PALETA_ASP_FIJA)
            col = "_categoria_snasp"

            n_rescatados = 0
            if col_snasp:
                n_rescatados = int((gdf[col_snasp].isna() & gdf[col].notna()).sum())

            n_sin_dato = gdf[col].isna().sum()
            gdf_valido = gdf[gdf[col].notna()]

            color_map = construir_mapa_colores(gdf_valido[col], PALETA_ASP_FIJA, PALETA_ASP_RESPALDO)
            style_fn = crear_style_categorico(color_map, col, fill=True)
            campos_tt = [c for c in [col, "Nombre", "NOMBRE", "Superficie", "SUPERFICIE"] if c in gdf_valido.columns]
            tooltip = folium.GeoJsonTooltip(fields=campos_tt) if campos_tt else folium.GeoJsonTooltip(fields=list(gdf_valido.columns[:-1]))

            if n_rescatados > 0:
                st.sidebar.caption(
                    f"✅ {nombre}: {n_rescatados} polígono(s) sin '{col_snasp}' se clasificaron a partir del nombre."
                )
            if n_sin_dato > 0:
                st.sidebar.caption(
                    f"⚠️ {nombre}: {n_sin_dato} polígono(s) sin categoría ni pista en el nombre no se muestran en el mapa."
                )
        else:
            gdf_valido = gdf
            color_map = {}
            style_fn = None
            tooltip = folium.GeoJsonTooltip(fields=list(gdf.columns[:-1]))

        if len(gdf_valido) > 0:
            folium.GeoJson(gdf_valido, name=f"🌳 {nombre}", style_function=style_fn, tooltip=tooltip).add_to(m)

        if color_map:
            secciones_leyenda.append(seccion_leyenda_categorica("Áreas Protegidas", color_map, icono="🌳", forma="area"))

    # ── Hidrografía ─────────────────────────────────────────────
    elif es_hidro:
        col_tipo = col_dren

        if col_tipo:
            style_fn = crear_style_por_diccionario(col_tipo, ESTILOS_TIPO_HIDRO)
            cols_disp = [c for c in [col_tipo, "Nombre", "NOMBRE"] if c and c in gdf.columns]
        else:
            style_fn = lambda f: {"color": ESTILOS_TIPO_HIDRO["default"]["color"], "weight": 1.5, "opacity": 0.9}
            cols_disp = list(gdf.columns[:-1])

        folium.GeoJson(
            gdf, name=f"💧 {nombre}", style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=cols_disp) if cols_disp else None,
        ).add_to(m)

        if col_tipo:
            tipos_presentes = sorted(gdf[col_tipo].dropna().unique().tolist())
            leyenda_hidro = {}
            for t in tipos_presentes:
                t_norm = quitar_tildes(str(t).lower())
                color = ESTILOS_TIPO_HIDRO["default"]["color"]
                for key, vals in ESTILOS_TIPO_HIDRO.items():
                    if key != "default" and key in t_norm:
                        color = vals["color"]
                        break
                leyenda_hidro[str(t)] = color
            secciones_leyenda.append(seccion_leyenda_categorica("Red Hídrica", leyenda_hidro, icono="💧", forma="linea"))

    # ── Vialidad / Accesos ──────────────────────────────────────
    elif es_vial:
        col_tipo = detectar_columna(gdf, ["TIPO", "tipo", "Tipo"])

        if col_tipo:
            style_fn = crear_style_por_diccionario(col_tipo, ESTILOS_TIPO_TRANSPORTE)
            tooltip = folium.GeoJsonTooltip(fields=[col_tipo])
        else:
            style_fn = lambda f: {"color": ESTILOS_TIPO_TRANSPORTE["default"]["color"], "weight": 2, "opacity": 0.9}
            tooltip = folium.GeoJsonTooltip(fields=list(gdf.columns[:-1]))

        folium.GeoJson(gdf, name=f"🛣️ {nombre}", style_function=style_fn, tooltip=tooltip).add_to(m)

    # ── Límite regional / contexto ───────────────────────────────
    elif es_limite:
        campos_tt = [c for c in ["Region", "REGION"] if c in gdf.columns]
        folium.GeoJson(
            gdf, name=f"🗺️ {nombre}",
            style_function=lambda f: {"color": "#333333", "weight": 2.5, "fillOpacity": 0, "dashArray": "6,4"},
            tooltip=folium.GeoJsonTooltip(fields=campos_tt) if campos_tt else None,
        ).add_to(m)

    # ── Atractivos turísticos (categórico por CATEGORIA, SERNATUR) ───────
    elif es_atractivo:
        col_cat = detectar_columna(gdf, ["CATEGORIA"])
        color_map = construir_mapa_colores(gdf[col_cat], PALETA_ATRACTIVOS_FIJA, PALETA_ATRACTIVOS_RESPALDO)

        def estilo_atractivo(feature, color_map=color_map, col_cat=col_cat):
            val = str(feature["properties"].get(col_cat, "")).strip()
            color = color_map.get(val, "#616161")
            return {"fillColor": color, "color": "#ffffff", "weight": 2,
                    "radius": 7, "fillOpacity": 0.95}

        campos_tt = [c for c in ["NOMBRE", "Nombre", "CATEGORIA", "TIPO", "COMUNA", "JERARQUIA"]
                     if c in gdf.columns]

        folium.GeoJson(
            gdf, name=f"📍 {nombre}",
            marker=folium.CircleMarker(radius=6, fill=True),
            style_function=estilo_atractivo,
            tooltip=folium.GeoJsonTooltip(fields=campos_tt) if campos_tt else None,
        ).add_to(m)

        secciones_leyenda.append(seccion_leyenda_categorica("Atractivos Turísticos", color_map, icono="📍", forma="pin"))

    # ── Capas de puntos genéricas (ej. sitios Ramsar, centros poblados) ─
    elif geom_base == "Point":
        campos_tt = list(gdf.columns[:-1])[:4]
        folium.GeoJson(
            gdf,
            name=f"📍 {nombre}",
            marker=folium.CircleMarker(radius=7, fill=True, fill_opacity=0.95,
                                        color="#ffffff", fill_color="#EC407A", weight=2),
            tooltip=folium.GeoJsonTooltip(fields=campos_tt) if campos_tt else None,
        ).add_to(m)

    # ── Resto ────────────────────────────────────────────────────
    else:
        folium.GeoJson(
            gdf, name=nombre,
            tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns[:-1])),
        ).add_to(m)

# ─────────────────────────────────────────────────────────────
# PASO 7: Leyenda (panel único), control de capas y render
# ─────────────────────────────────────────────────────────────

# Panel único de leyendas: nunca se encima ni se corta, tiene scroll propio.
panel_html = panel_leyendas_html(secciones_leyenda)
if panel_html:
    m.get_root().html.add_child(folium.Element(panel_html))

# El control de capas se mueve a bottomleft y colapsado para no chocar
# con el panel de leyendas (que ahora ocupa el topright).
folium.LayerControl(collapsed=True, position="bottomleft").add_to(m)

# Mapa responsive: se adapta al ancho del contenedor en vez de un
# tamaño fijo en píxeles (eso era lo que provocaba el corte a la derecha).
st_folium(m, use_container_width=True, height=700, returned_objects=[])

# ─────────────────────────────────────────────────────────────
# Info sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.write(f"Capas vectoriales: **{len(capas)}**")
st.sidebar.write(f"Rasters: **{len(rasters)}**")
st.sidebar.write(f"Activas: **{len(capas_activas) + len(rasters_activos)}**")
