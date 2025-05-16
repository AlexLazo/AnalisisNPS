import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
import warnings
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Configuración de la página mejorada
st.set_page_config(
    page_title="Dashboard NPS Predictivo Avanzado Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "# Dashboard NPS Predictivo v3.0 Pro"
    }
)

# =============================================
# ESTILOS MEJORADOS
# =============================================
st.markdown("""
<style>
    /* Tarjetas de métricas mejoradas */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        margin-bottom: 20px;
        border-left: 6px solid #1f77b4;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .metric-title {
        font-size: 16px;
        font-weight: 700;
        color: #555555;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #333333;
    }
    .metric-delta {
        font-size: 14px;
        font-weight: 600;
        margin-top: 8px;
        padding: 4px 8px;
        border-radius: 12px;
        display: inline-block;
    }
    .positive {
        background-color: #e8f5e9;
        color: #27ae60;
    }
    .negative {
        background-color: #ffebee;
        color: #e74c3c;
    }
    .neutral {
        background-color: #fff8e1;
        color: #f39c12;
    }
    
    /* Encabezados mejorados */
    .header-style {
        font-size: 26px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #1f77b4;
        background: linear-gradient(90deg, #f8f9fa, #e3f2fd);
        padding: 12px;
        border-radius: 8px;
    }
    .subheader-style {
        font-size: 20px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 16px;
        padding-left: 8px;
        border-left: 4px solid #1f77b4;
    }
    
    /* Tarjetas especiales mejoradas */
    .special-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 6px solid;
    }
    .seasonal-card {
        border-left-color: #6c757d;
    }
    .driver-card {
        border-left-color: #ffc107;
    }
    .risk-card {
        border-left-color: #e74c3c;
    }
    .opportunity-card {
        border-left-color: #2ecc71;
    }
    
    /* Selectores y controles mejorados */
    .stSelectbox, .stMultiselect, .stSlider, .stDateInput {
        margin-bottom: 18px;
    }
    .stSelectbox>div, .stMultiselect>div {
        border-radius: 10px !important;
    }
    
    /* Tabs mejorados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 12px 12px 0 0;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(31, 119, 180, 0.2);
    }
    .stTabs [aria-selected="false"]:hover {
        background-color: #e3f2fd !important;
    }
    
    /* Dataframes mejorados */
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Botones mejorados */
    .stButton>button {
        border-radius: 12px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        background-color: #1f77b4;
        color: white;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background-color: #1565c0;
    }
    .stDownloadButton>button {
        background-color: #2ecc71 !important;
    }
    .stDownloadButton>button:hover {
        background-color: #27ae60 !important;
    }
    
    /* Tooltips mejorados */
    .stTooltip {
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Spinners mejorados */
    .stSpinner>div {
        border: 4px solid rgba(31, 119, 180, 0.2);
        border-radius: 50%;
        border-top: 4px solid #1f77b4;
        width: 36px;
        height: 36px;
    }
    
    /* Mejoras generales */
    .stMarkdown {
        line-height: 1.6;
    }
    .css-1v3fvcr {
        padding: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# FUNCIONES UTILITARIAS MEJORADAS
# =============================================
def custom_metric(label, value, delta=None, delta_type=None, help_text=None, icon=None, unit=""):
    """Función mejorada para tarjetas de métricas con iconos y mejor formato"""
    delta_class = ""
    if delta_type == "positive":
        delta_class = "positive"
    elif delta_type == "negative":
        delta_class = "negative"
    elif delta_type == "neutral":
        delta_class = "neutral"
    
    icon_html = f"<span style='font-size:24px; margin-right:8px; vertical-align: middle;'>{icon}</span>" if icon else ""
    delta_html = f"<div class='metric-delta {delta_class}'>{delta}</div>" if delta else ""
    
    html = f"""
    <div class="metric-card">
        <div class="metric-title">{icon_html}{label}{'  ⓘ' if help_text else ''}</div>
        <div class="metric-value">{value}{unit}</div>
        {delta_html}
    </div>
    """
    
    if help_text:
        st.markdown(html, unsafe_allow_html=True)
        st.info(help_text, icon="ℹ️")
    else:
        st.markdown(html, unsafe_allow_html=True)

def calculate_nps(scores):
    """Calcula el NPS a partir de una serie de scores"""
    if len(scores) == 0:
        return 0
    promoters = (scores >= 9).sum()
    detractors = (scores <= 6).sum()
    return (promoters - detractors) / len(scores) * 100

def format_date_column(df, column_name):
    """Formatea columnas de fecha para mejor visualización"""
    if column_name in df.columns:
        df[column_name] = pd.to_datetime(df[column_name]).dt.strftime('%Y-%m-%d')
    return df

def calculate_frequency(df, id_col='customer_id', date_col='date'):
    """Calcula la frecuencia de encuestas por cliente"""
    if id_col not in df.columns or date_col not in df.columns:
        return None
    
    freq_df = df.groupby(id_col)[date_col].nunique().reset_index()
    freq_df.columns = [id_col, 'survey_frequency']
    return freq_df

# =============================================
# CARGAR DATOS CON CACHÉ MEJORADO
# =============================================
@st.cache_data(ttl=3600, show_spinner="Cargando y procesando datos...")
def load_data():
    try:
        # Cargar datos históricos con manejo de errores mejorado
        hist_df = pd.read_excel("Historico clientes Nps.xlsx", engine='openpyxl')
        hist_df.columns = hist_df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Procesamiento de fechas mejorado
        if 'date_(año)' in hist_df.columns and 'date_(mes)' in hist_df.columns:
            month_map = {
                'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
            }
            
            hist_df['date'] = hist_df.apply(
                lambda row: datetime(
                    year=int(float(row['date_(año)'])),
                    month=month_map.get(str(row['date_(mes)']).lower()[:3], 1),
                    day=1
                ) if not pd.isna(row['date_(año)']) else pd.NaT,
                axis=1
            )
            hist_df = hist_df.dropna(subset=['date'])
        
        # Renombrar columnas con mapeo más robusto
        column_mapping = {
            'poc_id': 'customer_id',
            'ddc_name': 'ddc_name',
            'pdv': 'pdv',
            'score': 'score',
            'cont': 'category',
            'secondary_driver': 'secondary_driver',
            'primary_driver': 'primary_driver',
            'sales_region': 'region',
            'sales_sub_region': 'sub_region',
            'segment_name': 'segment',
            'nps_type': 'nps_type',
            'region': 'region',
            'segment': 'segment'
        }
        hist_df = hist_df.rename(columns={k: v for k, v in column_mapping.items() if k in hist_df.columns})
        
        if 'score' in hist_df.columns:
            hist_df = hist_df[(hist_df['score'] >= 0) & (hist_df['score'] <= 10)]

        if 'score' in hist_df.columns:
            hist_df['category'] = pd.cut(
                hist_df['score'], 
                bins=[-1, 6, 8, 11], 
                labels=['Detractor', 'Passive', 'Promoter'],
                right=True
            )
        # Crear categorías si no existen con lógica mejorada
        if 'category' not in hist_df.columns and 'score' in hist_df.columns:
            hist_df['category'] = pd.cut(
                hist_df['score'], 
                bins=[-1, 6, 8, 11], 
                labels=['Detractor', 'Passive', 'Promoter'],
                right=True
            )
        
        # Asegurar que los scores estén entre 0 y 10
        if 'score' in hist_df.columns:
            hist_df = hist_df[(hist_df['score'] >= 0) & (hist_df['score'] <= 10)]
        
        # Calcular frecuencia de encuestas por cliente
        freq_df = calculate_frequency(hist_df)
        if freq_df is not None:
            hist_df = hist_df.merge(freq_df, on='customer_id', how='left')
        
        # Cargar predicciones con manejo de errores
        pred_df = pd.read_excel("predicciones_nps_completas.xlsx", engine='openpyxl')
        pred_df.columns = pred_df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Procesar fechas con manejo de errores
        date_columns = ['date', 'prediction_month', 'next_survey_date']
        date_columns = [col for col in date_columns if col in pred_df.columns]
        
        for col in date_columns:
            pred_df[col] = pd.to_datetime(pred_df[col], errors='coerce', format='mixed')
            # Eliminar filas con fechas inválidas
            pred_df = pred_df[~pd.isna(pred_df[col])]
        
        # Asegurar tipos de datos correctos para scores
        if 'predicted_score' in pred_df.columns:
            pred_df['predicted_score'] = pd.to_numeric(pred_df['predicted_score'], errors='coerce')
            pred_df = pred_df[~pd.isna(pred_df['predicted_score'])]
            
            # Asegurar que los scores predichos estén entre 0 y 10
            pred_df = pred_df[(pred_df['predicted_score'] >= 0) & (pred_df['predicted_score'] <= 10)]
            
            # Crear categorías predichas si no existen
            if 'predicted_category' not in pred_df.columns:
                pred_df['predicted_category'] = pd.cut(
                    pred_df['predicted_score'], 
                    bins=[-1, 6, 8, 11], 
                    labels=['Detractor', 'Passive', 'Promoter'],
                    right=True
                )
        
        return hist_df, pred_df
    
    except Exception as e:
        st.error(f"Error crítico al cargar datos: {str(e)}")
        st.error("Por favor verifica que los archivos estén en el formato correcto y en la ubicación esperada.")
        return None, None

# =============================================
# INTERFAZ DE USUARIO - SIDEBAR MEJORADO
# =============================================
st.sidebar.header("🔍 Filtros Avanzados")
st.sidebar.markdown("### Configuración de Visualización")

# Mostrar spinner mientras se cargan los datos
with st.spinner("Cargando y procesando datos, por favor espere..."):
    hist_df, pred_df = load_data()

if hist_df is None or pred_df is None:
    st.error("No se pudieron cargar los datos necesarios. Verifica los archivos y formatos.")
    st.stop()

# Filtro por DDC Name con búsqueda y selección inteligente
if 'ddc_name' in hist_df.columns:
    ddc_options = hist_df['ddc_name'].unique().tolist()
    selected_ddc = st.sidebar.multiselect(
        "Filtrar por DDC Name",
        options=ddc_options,
        default=ddc_options[:3] if len(ddc_options) > 3 else ddc_options,
        help="Selecciona uno o más DDC para filtrar"
    )

# Filtros dinámicos basados en columnas disponibles
available_filters = {
    'region': 'Región' if 'region' in hist_df.columns else None,
    'sub_region': 'Sub Región' if 'sub_region' in hist_df.columns else None,
    'segment': 'Segmento' if 'segment' in hist_df.columns else None,
    'primary_driver': 'Driver Principal' if 'primary_driver' in hist_df.columns else None,
    'nps_type': 'Tipo NPS' if 'nps_type' in hist_df.columns else None
}

filters = {}
for col, label in available_filters.items():
    if label and col in hist_df.columns:
        options = hist_df[col].dropna().unique().tolist()
        filters[col] = st.sidebar.multiselect(
            label, 
            options, 
            default=options[:3] if len(options) > 3 else options,
            help=f"Filtrar por {label.lower()}"
        )

if 'category' in hist_df.columns:
    # Normalizar categorías
    hist_df['category'] = (
        hist_df['category']
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            '0': 'Detractor',
            '1': 'Passive',
            '2': 'Promoter',
            '0.0': 'Detractor',
            '1.0': 'Passive',
            '2.0': 'Promoter',
            'detractor': 'Detractor',
            'passive': 'Passive',
            'promoter': 'Promoter'
        })
        .str.capitalize()
    )

    # Definir todas las categorías posibles
    valid_categories = ['Detractor', 'Passive', 'Promoter']
    available_categories = [cat for cat in valid_categories if cat in hist_df['category'].unique()]

    # Mostrar siempre las 3 opciones, pero deshabilitar las que no existan
    selected_categories = st.sidebar.multiselect(
        "Filtrar por Categoría NPS",
        options=valid_categories,
        default=available_categories,
        help="Selecciona las categorías NPS a incluir"
    )

# Filtro de frecuencia de encuestas por cliente
if 'survey_frequency' in hist_df.columns:
    freq_options = {
        "Todos": (0, float('inf')),
        "Una vez": (1, 1),
        "2-3 veces": (2, 3),
        "4-5 veces": (4, 5),
        "Más de 5 veces": (6, float('inf'))
    }
    
    selected_freq = st.sidebar.selectbox(
        "Frecuencia de Encuestas por Cliente",
        options=list(freq_options.keys()),
        index=0,
        help="Filtrar clientes por número de veces que han respondido encuestas"
    )
    min_freq, max_freq = freq_options[selected_freq]
else:
    min_freq, max_freq = 0, float('inf')

# Filtros de fecha mejorados con rangos inteligentes
if 'date' in hist_df.columns:
    min_date = hist_df['date'].min().to_pydatetime().date()
    max_date = hist_df['date'].max().to_pydatetime().date()
    
    # Rangos predefinidos útiles
    date_ranges = {
        "Últimos 3 meses": (max_date - timedelta(days=90), max_date),
        "Últimos 6 meses": (max_date - timedelta(days=180), max_date),
        "Último año": (max_date - timedelta(days=365), max_date),
        "Todo el histórico": (min_date, max_date),
        "Personalizado": None
    }
    
    selected_range = st.sidebar.selectbox(
        "Rango histórico predefinido",
        options=list(date_ranges.keys()),
        index=2,
        help="Selecciona un rango predefinido o personalizado"
    )
    
    if selected_range == "Personalizado":
        hist_date_range = st.sidebar.date_input(
            "Rango histórico personalizado",
            value=(max_date - timedelta(days=180), max_date),
            min_value=min_date,
            max_value=max_date,
            help="Selecciona el rango de fechas históricas a analizar"
        )
    else:
        hist_date_range = date_ranges[selected_range]

if 'prediction_month' in pred_df.columns:
    pred_min = pred_df['prediction_month'].min().to_pydatetime().date()
    pred_max = pred_df['prediction_month'].max().to_pydatetime().date()
    
    pred_date_range = st.sidebar.date_input(
        "Rango predicho",
        value=(pred_min, pred_max),
        min_value=pred_min,
        max_value=pred_max,
        help="Selecciona el rango de fechas predichas a visualizar"
    )

# Filtro de score predicho con valores inteligentes (0-10)
if 'predicted_score' in pred_df.columns:
    score_range = st.sidebar.slider(
        "Rango de Score Predicho (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=(0.0, 10.0),
        step=0.5,
        help="Filtra las predicciones por rango de score"
    )

# Selector de tema visual mejorado
theme = st.sidebar.selectbox(
    "Tema Visual",
    options=["Light", "Dark", "Plotly", "Seaborn", "GGplot"],
    index=0,
    help="Cambia el tema visual del dashboard"
)

# Selector de paleta de colores
color_palette = st.sidebar.selectbox(
    "Paleta de Colores",
    options=["Plotly", "D3", "Viridis", "Plasma", "Inferno", "Magma", "Cividis"],
    index=0,
    help="Selecciona una paleta de colores para las visualizaciones"
)

# =============================================
# FUNCIÓN DE FILTRADO MEJORADA
# =============================================
def apply_filters(df, is_historical=True):
    filtered_df = df.copy()
    
    # Aplicar filtro por DDC Name
    if 'ddc_name' in filtered_df.columns and 'selected_ddc' in globals() and selected_ddc:
        filtered_df = filtered_df[filtered_df['ddc_name'].isin(selected_ddc)]
    
    # Aplicar filtros de columnas
    for col, values in filters.items():
        if col in filtered_df.columns and values:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    
    # Aplicar filtro de categorías
    if is_historical and 'category' in filtered_df.columns and 'selected_categories' in globals() and selected_categories:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]
    
    # Aplicar filtro de frecuencia de encuestas
    if is_historical and 'survey_frequency' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['survey_frequency'] >= min_freq) & 
            (filtered_df['survey_frequency'] <= max_freq)
        ]
    
    # Aplicar filtro de fechas
    if is_historical and 'date' in filtered_df.columns and len(hist_date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= hist_date_range[0]) & 
            (filtered_df['date'].dt.date <= hist_date_range[1])
        ]
    elif not is_historical and 'prediction_month' in filtered_df.columns and len(pred_date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['prediction_month'].dt.date >= pred_date_range[0]) & 
            (filtered_df['prediction_month'].dt.date <= pred_date_range[1])
        ]
    
    # Aplicar filtro de score predicho (0-10)
    if not is_historical and 'predicted_score' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['predicted_score'] >= score_range[0]) & 
            (filtered_df['predicted_score'] <= score_range[1])
        ]
    
    return filtered_df

# Aplicar filtros con manejo de errores
try:
    hist_filtered = apply_filters(hist_df)
    pred_filtered = apply_filters(pred_df, False)
except Exception as e:
    st.error(f"Error al aplicar filtros: {str(e)}")
    st.stop()

# =============================================
# SECCIÓN 1: KPI PRINCIPALES MEJORADOS
# =============================================
st.markdown("## 📊 Métricas Clave NPS")
st.markdown("### Resumen de desempeño actual y predicciones")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # NPS Actual (calculado correctamente)
    if 'score' in hist_filtered.columns and not hist_filtered.empty:
        current_nps_value = calculate_nps(hist_filtered['score'])
        delta_type = "positive" if current_nps_value > 0 else "negative" if current_nps_value < 0 else "neutral"
        custom_metric(
            "NPS Actual", 
            f"{current_nps_value:.0f}", 
            "Positivo" if current_nps_value > 0 else "Negativo" if current_nps_value < 0 else "Neutral",
            delta_type,
            "Net Promoter Score calculado como % Promotores - % Detractores",
            "📈",
            ""
        )
    else:
        custom_metric("NPS Actual", "N/A", help_text="No hay datos de score disponibles", icon="⚠️")

with col2:
    # NPS Predicho
    if 'predicted_score' in pred_filtered.columns and not pred_filtered.empty:
        predicted_nps_value = calculate_nps(pred_filtered['predicted_score'])
        delta_value = predicted_nps_value - current_nps_value if 'current_nps_value' in locals() else 0
        delta_type = "positive" if delta_value > 0 else "negative" if delta_value < 0 else "neutral"
        custom_metric(
            "NPS Predicho", 
            f"{predicted_nps_value:.0f}", 
            f"{delta_value:+.0f} vs actual" if 'current_nps_value' in locals() else "",
            delta_type,
            "NPS predicho para el período seleccionado basado en el modelo",
            "🔮",
            ""
        )
    else:
        custom_metric("NPS Predicho", "N/A", help_text="No hay predicciones disponibles", icon="⚠️")

with col3:
    # Cantidad de Promotores y % Promotores
    if 'category' in hist_filtered.columns and not hist_filtered.empty:
        promoters = (hist_filtered['category'] == 'Promoter').sum()
        total = len(hist_filtered)
        promoter_percentage = (promoters / total) * 100 if total > 0 else 0
        
        delta_type = "positive" if promoter_percentage > 60 else "negative" if promoter_percentage < 40 else "neutral"
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🌟 Promotores</div>
            <div class="metric-value">{}</div>
            <div class="metric-value" style="font-size: 24px;">({:.1f}%)</div>
            <div class="metric-delta {}">{}</div>
        </div>
        """.format(
            promoters,
            promoter_percentage,
            delta_type,
            "Alto" if promoter_percentage > 60 else "Bajo" if promoter_percentage < 40 else "Moderado"
        ), unsafe_allow_html=True)
        
        st.info("Clientes con score 9-10 (Promotores)", icon="ℹ️")
    else:
        custom_metric("Promotores", "N/A", help_text="No hay datos de categorías disponibles", icon="⚠️")

with col4:
    # Cantidad de Detractores y % Detractores
    if 'category' in hist_filtered.columns and not hist_filtered.empty:

        # Asegura que no hay nulos y que el valor es exactamente 'Detractor'
        detractors = (hist_filtered['category'].fillna('').astype(str).str.strip() == 'Detractor').sum()
        total = len(hist_filtered)
        detractor_percentage = (detractors / total) * 100 if total > 0 else 0

        delta_type = "negative" if detractor_percentage > 20 else "positive" if detractor_percentage < 10 else "neutral"

        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">⚠️ Detractores</div>
            <div class="metric-value">{}</div>
            <div class="metric-value" style="font-size: 24px;">({:.1f}%)</div>
            <div class="metric-delta {}">{}</div>
        </div>
        """.format(
            detractors,
            detractor_percentage,
            delta_type,
            "Alto riesgo" if detractor_percentage > 20 else "Bajo riesgo" if detractor_percentage < 10 else "Riesgo moderado"
        ), unsafe_allow_html=True)

        st.info("Clientes con score 0-6 (Detractores)", icon="ℹ️")
    else:
        custom_metric("Detractores", "N/A", help_text="No hay datos de categorías disponibles", icon="⚠️")
# Segunda fila de métricas
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    # Frecuencia promedio de encuestas
    if 'survey_frequency' in hist_filtered.columns and not hist_filtered.empty:
        avg_freq = hist_filtered['survey_frequency'].mean()
        custom_metric(
            "Encuestas por Cliente", 
            f"{avg_freq:.1f}", 
            "Promedio histórico",
            "neutral",
            "Número promedio de veces que los clientes han respondido encuestas",
            "🔄",
            ""
        )

with col2:
    # Variación de score
    if 'score' in hist_filtered.columns and 'customer_id' in hist_filtered.columns and not hist_filtered.empty:
        score_variation = hist_filtered.groupby('customer_id')['score'].std().mean()
        custom_metric(
            "Variación Score", 
            f"{score_variation:.2f}", 
            "Promedio por cliente",
            "neutral" if score_variation < 2 else "negative",
            "Variación promedio en scores por cliente (desviación estándar)",
            "📊",
            ""
        )

with col3:
    # Tasa de respuesta
    if not hist_filtered.empty and 'predicted_score' in pred_filtered.columns and not pred_filtered.empty:
        response_rate = len(hist_filtered['customer_id'].unique()) / len(pred_filtered['customer_id'].unique()) * 100
        custom_metric(
            "Tasa de Respuesta", 
            f"{response_rate:.1f}", 
            "Histórico vs Base",
            "positive" if response_rate > 70 else "negative",
            "Porcentaje de clientes en base que han respondido al menos una encuesta",
            "✉️",
            "%"
        )

with col4:
    # Proporción de nuevos clientes
    if 'survey_frequency' in hist_filtered.columns and not hist_filtered.empty:
        new_customers = (hist_filtered['survey_frequency'] == 1).sum()
        total = len(hist_filtered)
        new_percentage = (new_customers / total) * 100 if total > 0 else 0
        
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">🆕 Clientes Nuevos</div>
            <div class="metric-value">{}</div>
            <div class="metric-value" style="font-size: 24px;">({:.1f}%)</div>
            <div class="metric-delta neutral">Primera encuesta</div>
        </div>
        """.format(new_customers, new_percentage), unsafe_allow_html=True)
        
        st.info("Clientes que han respondido solo una encuesta", icon="ℹ️")

# =============================================
# SECCIÓN 2: TENDENCIAS MEJORADAS
# =============================================
st.markdown("## 📈 Tendencias y Predicciones")
st.markdown("### Evolución histórica vs. proyecciones futuras")

tab1, tab2, tab3, tab4 = st.tabs(["Tendencia NPS", "Distribución Categorías", "Volumen Encuestas", "Análisis Detallado"])

with tab1:
    # Gráfico de tendencia NPS histórico vs predicho mejorado
    fig = go.Figure()
    
    # Datos históricos
    if not hist_filtered.empty and 'date' in hist_filtered.columns and 'score' in hist_filtered.columns:
        historical_nps = hist_filtered.groupby(hist_filtered['date'].dt.to_period('M')).apply(
            lambda x: calculate_nps(x['score'])
        ).reset_index(name='nps')
        historical_nps['date'] = historical_nps['date'].astype(str)
        
        fig.add_trace(go.Scatter(
            x=historical_nps['date'],
            y=historical_nps['nps'],
            name='NPS Histórico',
            line=dict(color='#3498db', width=4),
            mode='lines+markers',
            marker=dict(size=10, symbol='circle'),
            hovertemplate="<b>Fecha:</b> %{x}<br><b>NPS:</b> %{y:.1f}<extra></extra>"
        ))
    
    # Datos predichos
    if not pred_filtered.empty and 'prediction_month' in pred_filtered.columns and 'predicted_score' in pred_filtered.columns:
        predicted_nps = pred_filtered.groupby(pred_filtered['prediction_month'].dt.to_period('M')).apply(
            lambda x: calculate_nps(x['predicted_score'])
        ).reset_index(name='nps')
        predicted_nps['date'] = predicted_nps['prediction_month'].astype(str)
        
        fig.add_trace(go.Scatter(
            x=predicted_nps['date'],
            y=predicted_nps['nps'],
            name='NPS Predicho',
            line=dict(color='#e74c3c', width=4, dash='dot'),
            mode='lines+markers',
            marker=dict(size=10, symbol='diamond'),
            hovertemplate="<b>Fecha:</b> %{x}<br><b>NPS Predicho:</b> %{y:.1f}<extra></extra>"
        ))
    
    # Configuración del gráfico mejorado
    fig.update_layout(
        title='Evolución del NPS Histórico vs. Predicho',
        xaxis_title='Mes',
        yaxis_title='Net Promoter Score (NPS)',
        hovermode='x unified',
        template="plotly_white",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=50, r=50, b=50, t=80),
        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)')
    )
    
    # Línea de referencia para NPS positivo
    fig.add_hline(y=0, line_dash="dash", line_color="green", opacity=0.5, annotation_text="NPS Positivo", 
                 annotation_position="bottom right")
    
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    # Gráfico de distribución de categorías a lo largo del tiempo mejorado
    if not hist_filtered.empty and 'date' in hist_filtered.columns and 'category' in hist_filtered.columns:
        # Datos históricos
        hist_dist = hist_filtered.groupby([
            hist_filtered['date'].dt.to_period('M').astype(str),
            'category'
        ]).size().unstack(fill_value=0)
        
        # Normalizar a porcentajes
        hist_dist = hist_dist.div(hist_dist.sum(axis=1), axis=0) * 100

        # Verificar qué categorías están presentes y ordenarlas
        category_order = ['Promoter', 'Passive', 'Detractor']
        existing_categories = [cat for cat in category_order if cat in hist_dist.columns]
        if existing_categories:
            hist_dist = hist_dist[existing_categories]
        
        fig = go.Figure()
        
        # Colores basados en la paleta seleccionada
        colors = px.colors.qualitative.Plotly if color_palette == "Plotly" else getattr(px.colors.qualitative, color_palette)
        
        for i, col in enumerate(hist_dist.columns):
            fig.add_trace(go.Bar(
                x=hist_dist.index,
                y=hist_dist[col],
                name=col,
                marker_color=colors[i % len(colors)],
                hovertemplate="<b>Fecha:</b> %{x}<br><b>% {text}:</b> %{y:.1f}%<extra></extra>",
                text=[col]*len(hist_dist)
            ))

        fig.update_layout(
            barmode='stack',
            title='Distribución Histórica de Categorías NPS',
            xaxis_title='Mes',
            yaxis_title='Porcentaje',
            hovermode='x unified',
            template="plotly_white",
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos históricos de categorías disponibles")

with tab3:
    # Gráfico de volumen de encuestas mejorado
    fig = go.Figure()
    
    # Histórico
    if not hist_filtered.empty and 'date' in hist_filtered.columns:
        hist_volume = hist_filtered['date'].dt.to_period('M').value_counts().sort_index().reset_index()
        hist_volume.columns = ['date', 'count']
        hist_volume['date'] = hist_volume['date'].astype(str)
        
        fig.add_trace(go.Bar(
            x=hist_volume['date'],
            y=hist_volume['count'],
            name='Encuestas Históricas',
            marker_color='#3498db',
            opacity=0.8,
            hovertemplate="<b>Fecha:</b> %{x}<br><b>Encuestas:</b> %{y}<extra></extra>"
        ))
    
    # Predicciones
    if not pred_filtered.empty and 'prediction_month' in pred_filtered.columns:
        pred_volume = pred_filtered['prediction_month'].dt.to_period('M').value_counts().sort_index().reset_index()
        pred_volume.columns = ['date', 'count']
        pred_volume['date'] = pred_volume['date'].astype(str)
        
        fig.add_trace(go.Bar(
            x=pred_volume['date'],
            y=pred_volume['count'],
            name='Encuestas Predichas',
            marker_color='#e74c3c',
            opacity=0.8,
            hovertemplate="<b>Fecha:</b> %{x}<br><b>Encuestas Predichas:</b> %{y}<extra></extra>"
        ))
    
    fig.update_layout(
        title='Volumen de Encuestas Históricas vs. Predichas',
        xaxis_title='Mes',
        yaxis_title='Número de Encuestas',
        barmode='group',
        hovermode='x unified',
        template="plotly_white",
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    # Análisis detallado de tendencias con subplots
    if not hist_filtered.empty and 'date' in hist_filtered.columns and 'score' in hist_filtered.columns:
        # Preparar datos
        trend_data = hist_filtered.groupby(hist_filtered['date'].dt.to_period('M')).agg(
            nps=('score', calculate_nps),
            avg_score=('score', 'mean'),
            survey_count=('score', 'count'),
            promoters=('category', lambda x: (x == 'Promoter').sum()),
            detractors=('category', lambda x: (x == 'Detractor').sum())
        ).reset_index()
        trend_data['date'] = trend_data['date'].astype(str)
        
        # Crear subplots
        fig = make_subplots(rows=2, cols=2, 
                          subplot_titles=("Evolución del NPS", "Score Promedio", 
                                         "Promotores vs Detractores", "Correlación Score-Volumen"),
                          vertical_spacing=0.15, horizontal_spacing=0.1)
        
        # Gráfico 1: Evolución NPS
        fig.add_trace(
            go.Scatter(
                x=trend_data['date'],
                y=trend_data['nps'],
                name='NPS',
                line=dict(color='#3498db', width=3),
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        # Gráfico 2: Score Promedio
        fig.add_trace(
            go.Scatter(
                x=trend_data['date'],
                y=trend_data['avg_score'],
                name='Score Promedio',
                line=dict(color='#2ecc71', width=3),
                mode='lines+markers'
            ),
            row=1, col=2
        )
        
        # Gráfico 3: Promotores vs Detractores
        fig.add_trace(
            go.Bar(
                x=trend_data['date'],
                y=trend_data['promoters'],
                name='Promotores',
                marker_color='#27ae60'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=trend_data['date'],
                y=trend_data['detractors'],
                name='Detractores',
                marker_color='#e74c3c'
            ),
            row=2, col=1
        )
        
        # Gráfico 4: Correlación Score-Volumen
        fig.add_trace(
            go.Scatter(
                x=trend_data['survey_count'],
                y=trend_data['avg_score'],
                name='Correlación',
                mode='markers',
                marker=dict(
                    size=12,
                    color=trend_data['nps'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='NPS')
                ),
                text=trend_data['date']
            ),
            row=2, col=2
        )
        
        # Actualizar diseño
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white",
            title_text="Análisis Detallado de Tendencias",
            margin=dict(l=50, r=50, b=50, t=100),
            barmode='group'
        )
        
        # Actualizar ejes
        fig.update_yaxes(title_text="NPS", row=1, col=1)
        fig.update_yaxes(title_text="Score Promedio (0-10)", row=1, col=2)
        fig.update_yaxes(title_text="Cantidad", row=2, col=1)
        fig.update_yaxes(title_text="Score Promedio", row=2, col=2)
        fig.update_xaxes(title_text="Fecha", row=1, col=1)
        fig.update_xaxes(title_text="Fecha", row=1, col=2)
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_xaxes(title_text="Volumen Encuestas", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No hay datos suficientes para el análisis detallado")

# =============================================
# SECCIÓN 3: ANÁLISIS DE CLIENTES MEJORADO
# =============================================
st.markdown("## 🔍 Análisis Detallado de Clientes")

if not pred_filtered.empty:
    tab1, tab2, tab3 = st.tabs(["Clientes en Riesgo", "Oportunidades de Mejora", "Segmentación Avanzada"])
    
    with tab1:
        st.markdown("### 🔴 Clientes con Predicción de Detractores")
        
        # Filtro adicional para riesgo alto
        high_risk = pred_filtered[
            (pred_filtered['predicted_category'] == 'Detractor') &
            (pred_filtered['predicted_score'] <= 4)
        ]
        
        if not high_risk.empty:
            # Columnas base que siempre deberían estar
            base_cols = ['customer_id', 'ddc_name', 'predicted_score', 'predicted_category']
            
            # Columnas opcionales (solo se incluyen si existen en los datos)
            optional_cols = ['region', 'sub_region', 'segment', 'prediction_month', 'next_survey_date', 'primary_driver']
            
            # Filtrar solo las columnas que existen realmente en el DataFrame
            available_cols = [col for col in base_cols + optional_cols if col in high_risk.columns]
            
            # Mostrar métricas resumen
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Total Clientes en Riesgo**")
                st.markdown(f"<h2 style='text-align: center; color: #e74c3c;'>{len(high_risk)}</h2>", unsafe_allow_html=True)
            
            with col2:
                avg_score = high_risk['predicted_score'].mean()
                st.markdown("**Score Promedio**")
                st.markdown(f"<h2 style='text-align: center; color: #e74c3c;'>{avg_score:.1f}</h2>", unsafe_allow_html=True)
            
            with col3:
                if 'primary_driver' in high_risk.columns:
                    top_driver = high_risk['primary_driver'].mode()[0]
                    st.markdown("**Driver Principal**")
                    st.markdown(f"<h4 style='text-align: center; color: #e74c3c;'>{top_driver}</h4>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Si tenemos columnas adicionales, mostramos el dataframe
            if len(available_cols) > len(base_cols):
                st.dataframe(
                    high_risk[available_cols].sort_values('predicted_score'),
                    column_config={
                        "predicted_score": st.column_config.NumberColumn(
                            "Score Predicho",
                            format="%.1f",
                            help="Score predicho (0-10)"
                        ),
                        "predicted_category": st.column_config.TextColumn(
                            "Categoría",
                            help="Categoría NPS predicha"
                        ),
                        "next_survey_date": st.column_config.DateColumn(
                            "Próxima Encuesta",
                            format="YYYY-MM-DD"
                        )
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("Datos limitados disponibles para mostrar. Columnas faltantes: region, segment, etc.")
                
            # Análisis de drivers para clientes en riesgo
            if 'primary_driver' in high_risk.columns:
                st.markdown("#### Drivers Principales en Clientes de Alto Riesgo")
                driver_counts = high_risk['primary_driver'].value_counts().reset_index()
                driver_counts.columns = ['Driver', 'Clientes']
                
                fig = px.bar(
                    driver_counts.head(10),
                    x='Driver',
                    y='Clientes',
                    title="Top 10 Drivers en Clientes de Alto Riesgo",
                    color='Clientes',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Opción para exportar
            st.markdown("---")
            if st.button("📤 Exportar Lista de Clientes en Riesgo", key="export_high_risk"):
                csv = high_risk[available_cols].to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name="clientes_alto_riesgo.csv",
                    mime="text/csv"
                )
        else:
            st.success("🎉 No hay clientes identificados como alto riesgo con los filtros actuales")
    
    with tab2:
        st.markdown("### 🟡 Clientes Cerca de Cambiar Categoría")
        
        # Identificar clientes cerca del límite entre categorías
        borderline = pred_filtered[
            ((pred_filtered['predicted_score'] >= 7.5) & (pred_filtered['predicted_score'] <= 8.5)) |  # Cerca de ser Promotores
            ((pred_filtered['predicted_score'] >= 5.5) & (pred_filtered['predicted_score'] <= 6.5))    # Cerca de dejar de ser Detractores
        ]
        
        if not borderline.empty:
            # Mostrar métricas resumen
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Clientes en Límite**")
                st.markdown(f"<h2 style='text-align: center; color: #f39c12;'>{len(borderline)}</h2>", unsafe_allow_html=True)
            
            with col2:
                avg_score = borderline['predicted_score'].mean()
                st.markdown("**Score Promedio**")
                st.markdown(f"<h2 style='text-align: center; color: #f39c12;'>{avg_score:.1f}</h2>", unsafe_allow_html=True)
            
            with col3:
                if 'primary_driver' in borderline.columns:
                    top_driver = borderline['primary_driver'].mode()[0]
                    st.markdown("**Driver Principal**")
                    st.markdown(f"<h4 style='text-align: center; color: #f39c12;'>{top_driver}</h4>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            cols_to_show = [
                'customer_id', 'ddc_name',
                'predicted_score', 'predicted_category',
                'prediction_month', 'next_survey_date'
            ]
            cols_to_show = [col for col in cols_to_show if col in borderline.columns]
            
            st.dataframe(
                borderline[cols_to_show].sort_values('predicted_score'),
                column_config={
                    "predicted_score": st.column_config.ProgressColumn(
                        "Score Predicho",
                        format="%.1f",
                        min_value=0,
                        max_value=10,
                        help="Score predicho (0-10)"
                    ),
                    "predicted_category": st.column_config.TextColumn(
                        "Categoría",
                        help="Categoría NPS predicha"
                    ),
                    "next_survey_date": st.column_config.DateColumn(
                        "Próxima Encuesta",
                        format="YYYY-MM-DD"
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
            
            # Análisis de drivers para estos clientes
            if 'primary_driver' in borderline.columns:
                st.markdown("#### Drivers Principales en Clientes Cerca de Límites")
                
                tab1, tab2 = st.tabs(["Gráfico", "Tabla"])
                
                with tab1:
                    driver_counts = borderline['primary_driver'].value_counts().reset_index()
                    driver_counts.columns = ['Driver', 'Clientes']
                    
                    fig = px.bar(
                        driver_counts.head(10),
                        x='Driver',
                        y='Clientes',
                        title="Top 10 Drivers en Clientes Cerca de Cambio de Categoría",
                        color='Clientes',
                        color_continuous_scale='Oranges'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    driver_analysis = borderline.groupby('primary_driver').agg(
                        clientes=('customer_id', 'count'),
                        score_promedio=('predicted_score', 'mean'),
                        proxima_encuesta=('next_survey_date', lambda x: x.min().strftime('%Y-%m-%d') if not x.empty else 'N/A')
                    ).reset_index().sort_values('clientes', ascending=False)
                    
                    st.dataframe(
                        driver_analysis,
                        column_config={
                            "score_promedio": st.column_config.NumberColumn(
                                "Score Promedio",
                                format="%.2f"
                            )
                        },
                        hide_index=True,
                        use_container_width=True
                    )
            
            # Opción para exportar
            st.markdown("---")
            if st.button("📤 Exportar Lista de Oportunidades", key="export_borderline"):
                csv = borderline[cols_to_show].to_csv(index=False)
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name="clientes_oportunidades.csv",
                    mime="text/csv"
                )
        else:
            st.info("ℹ️ No hay clientes cerca de cambiar de categoría con los filtros actuales")
    
    with tab3:
        st.markdown("### 🔵 Segmentación Avanzada de Clientes")
        
        # Requiere datos históricos y de predicción
        if not hist_filtered.empty and not pred_filtered.empty:
            # Unir datos históricos y predicciones
            combined = pd.merge(
                hist_filtered,
                pred_filtered,
                on='customer_id',
                how='inner',
                suffixes=('_hist', '_pred')
            )
            
            # Seleccionar columnas relevantes
            segment_cols = [
                'customer_id', 'ddc_name_hist', 'region_hist', 'segment_hist',
                'score', 'category', 'primary_driver_hist',
                'predicted_score', 'predicted_category', 'prediction_month',
                'survey_frequency'
            ]
            segment_cols = [col for col in segment_cols if col in combined.columns]
            
            combined_segment = combined[segment_cols]
            
            # Análisis por segmento
            if 'segment_hist' in combined_segment.columns:
                st.markdown("#### Análisis por Segmento")
                
                segment_analysis = combined_segment.groupby('segment_hist').agg(
                    clientes=('customer_id', 'nunique'),
                    score_actual=('score', 'mean'),
                    score_predicho=('predicted_score', 'mean'),
                    cambio_score=('predicted_score', lambda x: x.mean() - combined_segment['score'].mean())
                ).reset_index().sort_values('cambio_score', ascending=False)
                
                fig = px.bar(
                    segment_analysis,
                    x='segment_hist',
                    y='cambio_score',
                    color='cambio_score',
                    color_continuous_scale='Bluered',
                    title="Cambio Predicho en Score por Segmento",
                    labels={'segment_hist': 'Segmento', 'cambio_score': 'Cambio en Score'},
                    hover_data=['score_actual', 'score_predicho']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if 'survey_frequency' in combined_segment.columns:
                st.markdown("#### Análisis por Frecuencia de Encuestas")
                
                # Verificar qué columnas están disponibles
                available_cols = combined_segment.columns.tolist()
                
                # Construir diccionario de agregación dinámicamente
                agg_dict = {'clientes': ('customer_id', 'nunique')}
                
                # Añadir score_actual solo si existe la columna 'score'
                if 'score' in available_cols:
                    agg_dict['score_actual'] = ('score', 'mean')
                
                # Añadir score_predicho solo si existe la columna 'predicted_score'
                if 'predicted_score' in available_cols:
                    agg_dict['score_predicho'] = ('predicted_score', 'mean')
                
                # Realizar el groupby con las columnas disponibles
                freq_analysis = combined_segment.groupby('survey_frequency').agg(**agg_dict).reset_index()
                
                # Crear gráfico solo si tenemos datos para mostrar
                fig = go.Figure()
                
                # Añadir traza para score_actual si está disponible
                if 'score_actual' in freq_analysis.columns:
                    fig.add_trace(go.Scatter(
                        x=freq_analysis['survey_frequency'],
                        y=freq_analysis['score_actual'],
                        name='Score Actual',
                        mode='lines+markers',
                        line=dict(color='blue', width=3)
                    ))
                
                # Añadir traza para score_predicho si está disponible
                if 'score_predicho' in freq_analysis.columns:
                    fig.add_trace(go.Scatter(
                        x=freq_analysis['survey_frequency'],
                        y=freq_analysis['score_predicho'],
                        name='Score Predicho',
                        mode='lines+markers',
                        line=dict(color='red', width=3, dash='dot')
                    ))
                
                # Solo actualizar el layout si hay trazas en el gráfico
                if len(fig.data) > 0:
                    fig.update_layout(
                        title='Relación entre Frecuencia de Encuestas y Score',
                        xaxis_title='Número de Encuestas',
                        yaxis_title='Score Promedio',
                        hovermode='x unified',
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar tabla completa
                st.markdown("#### Datos Completos de Segmentación")
                st.dataframe(
                    combined_segment,
                    hide_index=True,
                    use_container_width=True,
                    height=400
                )
            else:
                st.warning("Se necesitan ambos conjuntos de datos (histórico y predicciones) para el análisis de segmentación")

else:
    st.warning("No hay datos de predicciones disponibles para mostrar análisis de clientes")

# =============================================
# SECCIÓN 4: ANÁLISIS DE DRIVERS MEJORADO
# =============================================
if 'primary_driver' in hist_filtered.columns and 'category' in hist_filtered.columns:
    st.markdown("## 🚗 Análisis de Drivers de NPS")
    st.markdown("### Factores que influyen en la satisfacción del cliente")
    
    drivers_df = hist_filtered[hist_filtered['category'].isin(['Detractor', 'Passive'])]
    
    if not drivers_df.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["Distribución", "Impacto en Score", "Evolución Temporal", "Análisis Profundo"])
        
        with tab1:
            # Distribución de drivers
            driver_dist = drivers_df['primary_driver'].value_counts().reset_index()
            driver_dist.columns = ['Driver', 'Cantidad']
            
            fig = px.bar(
                driver_dist.head(20),
                x='Driver',
                y='Cantidad',
                title="Top 20 Drivers en Detractores/Pasivos",
                color='Cantidad',
                color_continuous_scale='Bluered',
                hover_data={'Driver': True, 'Cantidad': ':.0f'}
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Distribución por categoría
            if 'category' in drivers_df.columns:
                st.markdown("#### Distribución por Categoría NPS")
                cat_driver_dist = drivers_df.groupby(['primary_driver', 'category']).size().unstack().fillna(0)
                cat_driver_dist = cat_driver_dist.div(cat_driver_dist.sum(axis=1), axis=0) * 100
                cat_driver_dist = cat_driver_dist.sort_values('Detractor', ascending=False).head(20)
                
                fig = px.bar(
                    cat_driver_dist.reset_index(),
                    x='primary_driver',
                    y=['Detractor', 'Passive'],
                    title="Distribución de Categorías por Driver (Top 20)",
                    labels={'primary_driver': 'Driver', 'value': 'Porcentaje'},
                    barmode='stack',
                    color_discrete_sequence=['#e74c3c', '#f39c12']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Impacto en score
            driver_impact = drivers_df.groupby('primary_driver')['score'].agg(['mean', 'count', 'std']).reset_index()
            driver_impact.columns = ['Driver', 'Score Promedio', 'Cantidad', 'Desviación']
            driver_impact = driver_impact[driver_impact['Cantidad'] >= 5]  # Filtro de significancia
            
            fig = px.scatter(
                driver_impact.nlargest(30, 'Cantidad'),
                x='Score Promedio',
                y='Driver',
                size='Cantidad',
                color='Desviación',
                title="Impacto de Drivers en Score NPS (Top 30 por Volumen)",
                labels={'Score Promedio': 'Score Promedio (1-10)'},
                color_continuous_scale='Teal',
                hover_data={'Score Promedio': ':.2f', 'Cantidad': True, 'Desviación': ':.2f'}
            )
            fig.update_layout(height=700, yaxis={'categoryorder': 'total ascending'})
            fig.update_xaxes(range=[0, 10])
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlación entre frecuencia y score por driver
            if 'survey_frequency' in drivers_df.columns:
                st.markdown("#### Relación Frecuencia-Score por Driver")
                driver_freq = drivers_df.groupby('primary_driver').agg(
                    avg_score=('score', 'mean'),
                    avg_freq=('survey_frequency', 'mean'),
                    count=('score', 'count')
                ).reset_index()
                driver_freq = driver_freq[driver_freq['count'] >= 5]
                
                fig = px.scatter(
                    driver_freq,
                    x='avg_freq',
                    y='avg_score',
                    size='count',
                    color='avg_score',
                    hover_name='primary_driver',
                    title="Relación entre Frecuencia de Encuestas y Score por Driver",
                    labels={'avg_freq': 'Frecuencia Promedio', 'avg_score': 'Score Promedio'},
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Evolución temporal de drivers
            if 'date' in drivers_df.columns:
                time_drivers = drivers_df.groupby([
                    drivers_df['date'].dt.to_period('M').astype(str),
                    'primary_driver'
                ])['score'].mean().unstack().fillna(0)
                
                # Seleccionar top 5 drivers para visualización
                top_drivers = drivers_df['primary_driver'].value_counts().head(5).index.tolist()
                time_drivers = time_drivers[top_drivers]
                
                fig = go.Figure()
                for driver in top_drivers:
                    fig.add_trace(go.Scatter(
                        x=time_drivers.index,
                        y=time_drivers[driver],
                        name=driver,
                        mode='lines+markers',
                        line=dict(width=3)
                    ))
                
                fig.update_layout(
                    title='Evolución Temporal de Scores por Driver Principal (Top 5)',
                    xaxis_title='Mes',
                    yaxis_title='Score Promedio',
                    hovermode='x unified',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Heatmap de drivers por mes
                st.markdown("#### Heatmap de Drivers por Mes")
                time_driver_counts = drivers_df.groupby([
                    drivers_df['date'].dt.to_period('M').astype(str),
                    'primary_driver'
                ]).size().unstack().fillna(0)
                
                # Seleccionar top 10 drivers
                top_drivers = drivers_df['primary_driver'].value_counts().head(10).index.tolist()
                time_driver_counts = time_driver_counts[top_drivers]
                
                fig = px.imshow(
                    time_driver_counts.T,
                    labels=dict(x="Mes", y="Driver", color="Cantidad"),
                    aspect="auto",
                    color_continuous_scale='YlOrRd'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos de fechas para análisis temporal")
        
        with tab4:
            # Análisis profundo de drivers seleccionados
            selected_driver = st.selectbox(
                "Selecciona un Driver para análisis detallado",
                options=drivers_df['primary_driver'].value_counts().index.tolist()
            )
            
            if selected_driver:
                driver_data = drivers_df[drivers_df['primary_driver'] == selected_driver]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Estadísticas para {selected_driver}**")
                    
                    stats = driver_data['score'].describe().to_frame().T
                    stats['count'] = stats['count'].astype(int)
                    
                    st.dataframe(
                        stats,
                        column_config={
                            "count": "Casos",
                            "mean": "Promedio",
                            "std": "Desviación",
                            "min": "Mínimo",
                            "25%": "25% Percentil",
                            "50%": "Mediana",
                            "75%": "75% Percentil",
                            "max": "Máximo"
                        },
                        hide_index=True
                    )
                
                with col2:
                    st.markdown("**Distribución de Scores**")
                    
                    fig = px.histogram(
                        driver_data,
                        x='score',
                        nbins=10,
                        title=f"Distribución de Scores para {selected_driver}",
                        labels={'score': 'Score NPS'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Evolución temporal para el driver seleccionado
                if 'date' in driver_data.columns:
                    st.markdown("**Evolución Temporal**")
                    
                    driver_time = driver_data.groupby(driver_data['date'].dt.to_period('M')).agg(
                        avg_score=('score', 'mean'),
                        count=('score', 'count')
                    ).reset_index()
                    driver_time['date'] = driver_time['date'].astype(str)
                    
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=driver_time['date'],
                            y=driver_time['avg_score'],
                            name='Score Promedio',
                            line=dict(color='blue', width=3)
                        ),
                        secondary_y=False
                    )

                    fig.add_trace(
                        go.Bar(
                            x=driver_time['date'],
                            y=driver_time['count'],
                            name='Casos',
                            opacity=0.3,
                            marker_color='gray'
                        ),
                        secondary_y=True
                    )
                    
                    fig.update_layout(
                        title=f"Evolución de {selected_driver}",
                        xaxis_title='Mes',
                        hovermode='x unified'
                    )
                    
                    fig.update_yaxes(title_text="Score Promedio", secondary_y=False)
                    fig.update_yaxes(title_text="Número de Casos", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Segmentación del driver seleccionado
                if 'segment' in driver_data.columns:
                    st.markdown("**Análisis por Segmento**")
                    
                    driver_segment = driver_data.groupby('segment').agg(
                        avg_score=('score', 'mean'),
                        count=('score', 'count')
                    ).reset_index().sort_values('avg_score')
                    
                    fig = px.bar(
                        driver_segment,
                        x='segment',
                        y='avg_score',
                        color='count',
                        title=f"Score Promedio por Segmento - {selected_driver}",
                        labels={'avg_score': 'Score Promedio', 'segment': 'Segmento'},
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos suficientes para análisis de drivers con los filtros actuales")

# =============================================
# SECCIÓN 5: EXPORTACIÓN MEJORADA
# =============================================
st.markdown("## 📤 Exportación de Reportes")
st.markdown("### Genera informes personalizados basados en los filtros aplicados")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Reporte de Predicciones")
    if not pred_filtered.empty:
        with st.expander("🔽 Opciones de exportación", expanded=False):
            export_cols = st.multiselect(
                "Selecciona columnas a exportar",
                options=pred_filtered.columns,
                default=[
                    'customer_id', 'ddc_name',
                    'predicted_score', 'predicted_category',
                    'prediction_month', 'next_survey_date'
                ],
                key="pred_export_cols"
            )
            
            format_type = st.radio(
                "Formato de exportación",
                options=['Excel', 'CSV'],
                index=0,
                key="pred_format"
            )
            
            report_name = st.text_input(
                "Nombre del reporte",
                value="predicciones_nps",
                key="pred_report_name"
            )
            
            if st.button("🔄 Generar Reporte", key="export_predictions"):
                with st.spinner("Generando archivo..."):
                    export_data = pred_filtered[export_cols]
                    
                    if format_type == 'Excel':
                        with pd.ExcelWriter(f'{report_name}.xlsx') as writer:
                            export_data.to_excel(writer, index=False, sheet_name='Predicciones_NPS')
                            
                            # Agregar resumen estadístico
                            stats = export_data.describe(include='all').T
                            stats.to_excel(writer, sheet_name='Resumen_Estadistico')
                        
                        with open(f'{report_name}.xlsx', 'rb') as f:
                            st.download_button(
                                label="⬇️ Descargar Excel",
                                data=f,
                                file_name=f'{report_name}.xlsx',
                                mime='application/vnd.ms-excel'
                            )
                    else:
                        csv = export_data.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Descargar CSV",
                            data=csv,
                            file_name=f'{report_name}.csv',
                            mime='text/csv'
                        )
    else:
        st.warning("No hay predicciones para exportar")

with col2:
    st.markdown("#### Reporte Histórico")
    if not hist_filtered.empty:
        with st.expander("🔽 Opciones de exportación", expanded=False):
            hist_export_cols = st.multiselect(
                "Selecciona columnas históricas",
                options=hist_filtered.columns,
                default=[
                    'customer_id', 'ddc_name', 
                    'score', 'category', 'date'
                ],
                key="hist_export_cols"
            )
            
            hist_format = st.radio(
                "Formato histórico",
                options=['Excel', 'CSV'],
                index=0,
                key="hist_format"
            )
            
            hist_report_name = st.text_input(
                "Nombre del reporte histórico",
                value="historico_nps",
                key="hist_report_name"
            )
            
            if st.button("🔄 Generar Reporte", key="export_historical"):
                with st.spinner("Generando archivo..."):
                    hist_export_data = hist_filtered[hist_export_cols]
                    
                    if hist_format == 'Excel':
                        with pd.ExcelWriter(f'{hist_report_name}.xlsx') as writer:
                            hist_export_data.to_excel(writer, index=False, sheet_name='Historico_NPS')
                            
                            # Agregar resumen estadístico
                            stats = hist_export_data.describe(include='all').T
                            stats.to_excel(writer, sheet_name='Resumen_Estadistico')
                            
                            # Agregar análisis de categorías
                            if 'category' in hist_export_data.columns:
                                cat_analysis = hist_export_data['category'].value_counts().reset_index()
                                cat_analysis.columns = ['Categoria', 'Conteo']
                                cat_analysis.to_excel(writer, sheet_name='Analisis_Categorias', index=False)
                        
                        with open(f'{hist_report_name}.xlsx', 'rb') as f:
                            st.download_button(
                                label="⬇️ Descargar Excel",
                                data=f,
                                file_name=f'{hist_report_name}.xlsx',
                                mime='application/vnd.ms-excel'
                            )
                    else:
                        csv = hist_export_data.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Descargar CSV",
                            data=csv,
                            file_name=f'{hist_report_name}.csv',
                            mime='text/csv'
                        )
    else:
        st.warning("No hay datos históricos para exportar")

with col3:
    st.markdown("#### Reporte Personalizado")
    st.markdown("Crea un informe combinado con datos históricos y predicciones")
    
    if not hist_filtered.empty and not pred_filtered.empty:
        with st.expander("🔽 Opciones de exportación", expanded=False):
            combined_report_name = st.text_input(
                "Nombre del reporte combinado",
                value="reporte_nps_combinado",
                key="combined_report_name"
            )
            
            include_stats = st.checkbox(
                "Incluir análisis estadístico",
                value=True,
                key="include_stats"
            )
            
            include_drivers = st.checkbox(
                "Incluir análisis de drivers",
                value=True,
                key="include_drivers"
            )
            
            if st.button("🔄 Generar Reporte Combinado", key="export_combined"):
                with st.spinner("Combinando datos y generando reporte..."):
                    # Unir datos históricos y predicciones
                    combined = pd.merge(
                        hist_filtered,
                        pred_filtered,
                        on='customer_id',
                        how='inner',
                        suffixes=('_hist', '_pred')
                    )
                    
                    # Seleccionar columnas relevantes
                    combined_cols = [
                        'customer_id', 'ddc_name_hist', 'region_hist', 'segment_hist',
                        'score', 'category', 'primary_driver_hist',
                        'predicted_score', 'predicted_category', 'prediction_month',
                        'survey_frequency'
                    ]
                    combined_cols = [col for col in combined_cols if col in combined.columns]
                    
                    combined_report = combined[combined_cols]
                    
                    # Generar archivo
                    with pd.ExcelWriter(f'{combined_report_name}.xlsx') as writer:
                        combined_report.to_excel(writer, sheet_name='Resumen', index=False)
                        hist_filtered.to_excel(writer, sheet_name='Datos_Historicos', index=False)
                        pred_filtered.to_excel(writer, sheet_name='Predicciones', index=False)
                        
                        # Agregar hojas adicionales según opciones
                        if include_stats:
                            stats = combined_report.describe(include='all').T
                            stats.to_excel(writer, sheet_name='Estadisticas')
                            
                            # Análisis de cambios
                            if 'score' in combined_report.columns and 'predicted_score' in combined_report.columns:
                                combined_report['cambio_score'] = combined_report['predicted_score'] - combined_report['score']
                                change_stats = combined_report['cambio_score'].describe().to_frame().T
                                change_stats.to_excel(writer, sheet_name='Cambio_Score')
                        
                        if include_drivers and 'primary_driver_hist' in combined_report.columns:
                            driver_analysis = combined_report.groupby('primary_driver_hist').agg(
                                count=('customer_id', 'count'),
                                avg_score=('score', 'mean'),
                                avg_predicted=('predicted_score', 'mean'),
                                avg_change=('cambio_score', 'mean')
                            ).reset_index().sort_values('count', ascending=False)
                            driver_analysis.to_excel(writer, sheet_name='Analisis_Drivers', index=False)
                    
                    with open(f'{combined_report_name}.xlsx', 'rb') as f:
                        st.download_button(
                            label="⬇️ Descargar Reporte Completo",
                            data=f,
                            file_name=f'{combined_report_name}.xlsx',
                            mime='application/vnd.ms-excel'
                        )
    else:
        st.warning("Se necesitan ambos conjuntos de datos para generar reporte combinado")

# =============================================
# PIE DE PÁGINA MEJORADO
# =============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 14px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
    <p style="font-size: 16px; font-weight: bold; margin-bottom: 5px;">Dashboard NPS Predictivo v3.0 Pro</p>
    <p style="margin-bottom: 5px;">Última actualización: {}</p>
    <p style="margin-bottom: 5px;">© 2023 Equipo de Analítica de Clientes | Todos los derechos reservados</p>
    <p style="font-size: 12px; margin-top: 10px;">Este dashboard fue creado con Streamlit, Plotly y Pandas</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)