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

st.set_page_config(page_title="GeoVisualizador de Biodiversidad - Aysén", layout="wide")
st.title("🌳 GeoVisualizador de Biodiversidad y Áreas Protegidas — Región de Aysén")
st.write(
    "Aplicación desarrollada con Streamlit. Explora las áreas silvestres protegidas, "
    "la red hídrica y el relieve de la Región de Aysén del General Carlos Ibáñez del Campo. "
    "Selecciona las capas en el panel lateral y usa el Panel de Análisis para explorar estadísticas."
)

DATA = Path("data")
CRS_METRICO = None  # se calcula dinámicamente por capa con estimate_utm_crs()

import unicodedata

def quitar_tildes(texto):
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

# ─────────────────────────────────────────────────────────────
# PASO 1: Paletas cartográficas
# ─────────────────────────────────────────────────────────────

# Áreas protegidas: colores inspirados en categorías SNASPE/SNAP
PALETA_ASP_FIJA = {
    "parque":      "#1B5E20",
    "reserva":     "#66BB6A",
    "monumento":   "#F9A825",
    "santuario":   "#00838F",
    "biosfera":    "#6A1B9A",
    "ramsar":      "#0277BD",
    "prioritario": "#AD1457",
}
PALETA_ASP_RESPALDO = [
    "#2E7D32", "#558B2F", "#00695C", "#4527A0", "#EF6C00",
    "#00838F", "#6D4C41", "#C62828", "#5E35B1", "#F9A825",
]

# Vialidad / accesos: jerarquía vial estándar
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

# Hidrografía
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

# Atractivos turísticos: categorías SERNATUR
PALETA_ATRACTIVOS_FIJA = {
    "sitios naturales":            "#2E7D32",
    "museos manifestaciones":      "#6A1B9A",
    "folklore":                    "#EF6C00",
    "realizaciones tecnicas":      "#0277BD",
    "rutas y circuitos":           "#AD1457",
    "acontecimientos programados": "#F9A825",
}
PALETA_ATRACTIVOS_RESPALDO = ["#455A64", "#00838F", "#795548", "#C62828"]

# DEM: colormap hipsométrico (costa -> cordillera, coherente con Los Ríos)
COLORMAP_DEM = [
    (0.00, "#08306B"),   # azul profundo (nivel del mar / cuerpos de agua)
    (0.05, "#1B5E20"),   # verde oscuro - tierras bajas / bosque
    (0.20, "#4CAF50"),
    (0.40, "#AED581"),
    (0.60, "#DAA520"),
    (0.75, "#8B4513"),
    (0.90, "#D2B48C"),
    (1.00, "#FFFAFA"),   # blanco - cumbres andinas
]

# ─────────────────────────────────────────────────────────────
# PASO 2: Funciones de color dinámico y estilos
# ─────────────────────────────────────────────────────────────

def construir_mapa_colores(serie, paleta_fija=None, paleta_respaldo=None):
    """
    Asigna color a cada valor único de una serie.
    Usa primero coincidencias en paleta_fija (por palabra clave, case-insensitive),
    y para el resto reparte colores de paleta_respaldo.
    """
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
        color_map[str(v)] = color_asignado
    return color_map


def crear_style_categorico(color_map, col, fill=True, weight=0.8):
    def style_fn(feature):
        val = str(feature["properties"].get(col, ""))
        color = color_map.get(val, "#AAAAAA")
        if fill:
            return {"fillColor": color, "color": "#333333", "weight": weight, "fillOpacity": 0.65}
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
# PASO 3: Leyendas HTML
# ─────────────────────────────────────────────────────────────

def leyenda_categorica_html(titulo, color_map, icono="🔲", posicion_top="10px", posicion_right="10px"):
    items = ""
    for etiqueta, color in sorted(color_map.items()):
        items += f"""
        <div style="display:flex;align-items:center;margin:3px 0;">
          <div style="background:{color};width:16px;height:16px;
                      border:1px solid #555;margin-right:7px;
                      border-radius:2px;flex-shrink:0;"></div>
          <span style="font-size:11px;color:#222;">{etiqueta}</span>
        </div>"""
    return f"""
    <div style="
        position: fixed;
        top:   {posicion_top};
        right: {posicion_right};
        z-index: 1000;
        background: rgba(255,255,255,0.93);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #bbb;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
        max-height: 280px;
        overflow-y: auto;
        min-width: 170px;
        font-family: Arial, sans-serif;">
      <b style="font-size:12px;">{icono} {titulo}</b>
      <hr style="margin:5px 0;border-color:#ddd;">
      {items}
    </div>"""


def leyenda_dem_html(dem_min, dem_max, posicion_top="10px", posicion_right="10px"):
    stops = ", ".join([f"{color} {int(pct*100)}%" for pct, color in COLORMAP_DEM])
    gradient = f"linear-gradient(to top, {stops})"
    return f"""
    <div style="
        position: fixed;
        top:   {posicion_top};
        right: {posicion_right};
        z-index: 1000;
        background: rgba(255,255,255,0.93);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #bbb;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.25);
        min-width: 130px;
        font-family: Arial, sans-serif;">
      <b style="font-size:12px;">🏔️ Elevación (m)</b>
      <hr style="margin:5px 0;border-color:#ddd;">
      <div style="display:flex;align-items:stretch;gap:8px;">
        <div style="width:22px;height:150px;background:{gradient};
                    border:1px solid #888;border-radius:3px;flex-shrink:0;"></div>
        <div style="display:flex;flex-direction:column;
                    justify-content:space-between;font-size:11px;color:#333;">
          <span><b>{int(dem_max)} m</b></span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.75)} m</span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.50)} m</span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.25)} m</span>
          <span><b>{int(dem_min)} m</b></span>
        </div>
      </div>
    </div>"""

# ─────────────────────────────────────────────────────────────
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

capas = {}
for archivo in archivos_vec:
    nombre = archivo.stem.replace("_", " ")
    try:
        gdf = gpd.read_file(archivo)
        if gdf.crs is not None:
            gdf = gdf.to_crs(4326)
        capas[nombre] = gdf
    except Exception as e:
        st.warning(f"No fue posible cargar {archivo.name}: {e}")

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
    """Busca la primera columna existente entre una lista de nombres candidatos (case-insensitive)."""
    cols_low = {c.lower(): c for c in gdf.columns}
    for cand in candidatos:
        if cand.lower() in cols_low:
            return cols_low[cand.lower()]
    return None

def columnas_categoricas(gdf, max_categorias=25):
    """Columnas de tipo texto/objeto con un número razonable de categorías, útiles para filtrar/graficar."""
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

    # ── Estadísticas descriptivas ──
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

    # ── Filtro interactivo ──
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

    # ── Gráfico estadístico ──
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

leyendas_html = []
offset_top = 10

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
            leyendas_html.append(
                leyenda_dem_html(dem_min, dem_max, posicion_top=f"{offset_top}px", posicion_right="10px")
            )
            offset_top += 220
    except Exception as e:
        st.warning(f"No fue posible cargar raster '{nombre}': {e}")

# ─────────────────────────────────────────────────────────────
# PASO 6: Agregar vectores con estilos, filtro y leyendas
# ─────────────────────────────────────────────────────────────

for nombre in capas_activas:
    gdf = capas[nombre]
    nombre_low = nombre.lower()

    # Aplicar filtro interactivo si corresponde a la capa en análisis
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
        col = col_snasp

        if col:
            color_map = construir_mapa_colores(gdf[col], PALETA_ASP_FIJA, PALETA_ASP_RESPALDO)
            style_fn = crear_style_categorico(color_map, col, fill=True)
            campos_tt = [c for c in [col, "Nombre", "NOMBRE", "Superficie", "SUPERFICIE"] if c in gdf.columns]
            tooltip = folium.GeoJsonTooltip(fields=campos_tt) if campos_tt else folium.GeoJsonTooltip(fields=list(gdf.columns[:-1]))
        else:
            color_map = {}
            style_fn = None
            tooltip = folium.GeoJsonTooltip(fields=list(gdf.columns[:-1]))

        folium.GeoJson(gdf, name=f"🌳 {nombre}", style_function=style_fn, tooltip=tooltip).add_to(m)

        if color_map:
            leyendas_html.append(
                leyenda_categorica_html("Áreas Protegidas", color_map, icono="🌳",
                                         posicion_top=f"{offset_top}px", posicion_right="10px")
            )
            offset_top += min(60 + len(color_map) * 23, 300) + 10

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

        color_map_hidro = {}
        # Leyenda simple para hidrografía a partir de los tipos presentes
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
            leyendas_html.append(
                leyenda_categorica_html("Red Hídrica", leyenda_hidro, icono="💧",
                                         posicion_top=f"{offset_top}px", posicion_right="10px")
            )
            offset_top += min(60 + len(leyenda_hidro) * 23, 300) + 10

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
            val = str(feature["properties"].get(col_cat, ""))
            color = color_map.get(val, "#616161")
            return {"fillColor": color, "color": "#333333", "weight": 1,
                    "radius": 6, "fillOpacity": 0.85}

        campos_tt = [c for c in ["NOMBRE", "Nombre", "CATEGORIA", "TIPO", "COMUNA", "JERARQUIA"]
                     if c in gdf.columns]

        folium.GeoJson(
            gdf, name=f"📍 {nombre}",
            marker=folium.CircleMarker(radius=6, fill=True),
            style_function=estilo_atractivo,
            tooltip=folium.GeoJsonTooltip(fields=campos_tt) if campos_tt else None,
        ).add_to(m)

        leyendas_html.append(
            leyenda_categorica_html("Atractivos Turísticos", color_map, icono="📍",
                                     posicion_top=f"{offset_top}px", posicion_right="10px")
        )
        offset_top += min(60 + len(color_map) * 23, 300) + 10

    # ── Capas de puntos genéricas (ej. sitios Ramsar, centros poblados) ─
    elif geom_base == "Point":
        campos_tt = list(gdf.columns[:-1])[:4]
        folium.GeoJson(
            gdf,
            name=f"📍 {nombre}",
            marker=folium.CircleMarker(radius=6, fill=True, fill_opacity=0.85,
                                        color="#AD1457", fill_color="#EC407A", weight=1.5),
            tooltip=folium.GeoJsonTooltip(fields=campos_tt) if campos_tt else None,
        ).add_to(m)

    # ── Resto ────────────────────────────────────────────────────
    else:
        folium.GeoJson(
            gdf, name=nombre,
            tooltip=folium.GeoJsonTooltip(fields=list(gdf.columns[:-1])),
        ).add_to(m)

# ─────────────────────────────────────────────────────────────
# PASO 7: Leyendas + control de capas + render
# ─────────────────────────────────────────────────────────────

for html in leyendas_html:
    m.get_root().html.add_child(folium.Element(html))

folium.LayerControl(collapsed=False).add_to(m)

st_folium(m, width=1200, height=700)

# ─────────────────────────────────────────────────────────────
# Info sidebar
# ─────────────────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.write(f"Capas vectoriales: **{len(capas)}**")
st.sidebar.write(f"Rasters: **{len(rasters)}**")
st.sidebar.write(f"Activas: **{len(capas_activas) + len(rasters_activos)}**")
