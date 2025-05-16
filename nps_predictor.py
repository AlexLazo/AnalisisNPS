import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')

PREDICTION_START = datetime(2025, 1, 1)
PREDICTION_END = datetime(2026, 12, 31)

def parse_custom_date(year_col, month_col):
    month_map = {
        'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    try:
        if pd.isna(year_col) or pd.isna(month_col):
            return pd.NaT
        year = int(float(year_col))
        month_str = str(month_col).lower().strip()[:3]
        month = month_map.get(month_str, month_str)
        if isinstance(month, str):
            try:
                month = int(float(month_str))
            except:
                month = 1
        return datetime(year, month, 1)
    except Exception as e:
        print(f"Error parsing date: {year_col}-{month_col}. Error: {str(e)}")
        return pd.NaT

def load_data(filepath):
    try:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.strip().str.lower()
        column_mapping = {
            'poc id': 'customer_id',
            'ddc_name': 'ddc_name',
            'date (año)': 'year_part',
            'date (mes)': 'month_part',
            'pdv': 'pdv',
            'score': 'score',
            'cont': 'category',
            'primary': 'primary_driver',
            'secondary': 'secondary_driver'
        }
        for orig, new in column_mapping.items():
            if orig in df.columns:
                df = df.rename(columns={orig: new})
        if all(col in df.columns for col in ['year_part', 'month_part']):
            df['date'] = df.apply(
                lambda row: parse_custom_date(row['year_part'], row['month_part']), 
                axis=1
            )
        else:
            raise ValueError("No se encontraron las columnas necesarias 'date (año)' y 'date (mes)'")
        df = df.dropna(subset=['date'])
        if 'score' in df.columns:
            df['category'] = pd.cut(df['score'], bins=[-1, 6, 8, 11],
                                  labels=['Detractor', 'Passive', 'Promoter'])
        for driver in ['primary_driver', 'secondary_driver']:
            if driver not in df.columns:
                df[driver] = 'Unknown'
        df = df.sort_values(['customer_id', 'date']).drop_duplicates()
        if 'score' in df.columns:
            df = df[df['score'].between(0, 10)]
        return df
    except Exception as e:
        print(f"\n❌ Error al cargar datos: {str(e)}")
        return None

def create_features(df):
    df = df.sort_values(['customer_id', 'date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['days_since_last_survey'] = df.groupby('customer_id')['date'].diff().dt.days
    df['survey_frequency'] = 365 / df.groupby('customer_id')['date'].transform('count')
    for window in [3, 6, 12]:
        df[f'score_ma_{window}'] = df.groupby('customer_id')['score'].transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f'score_std_{window}'] = df.groupby('customer_id')['score'].transform(
            lambda x: x.rolling(window, min_periods=1).std())
    df['historical_promoter_rate'] = df.groupby('customer_id')['category'].transform(
        lambda x: (x == 'Promoter').expanding().mean())
    df['last_category'] = df.groupby('customer_id')['category'].shift(1)
    for driver_col in ['primary_driver', 'secondary_driver']:
        if driver_col in df.columns:
            freq_encoder = df[driver_col].value_counts(normalize=True)
            df[f'{driver_col}_freq'] = df[driver_col].map(freq_encoder)
            target_encoder = df.groupby(driver_col)['score'].mean()
            df[f'{driver_col}_target'] = df[driver_col].map(target_encoder)
    try:
        ts_data = df.groupby('date')['score'].mean().resample('MS').mean()
        decomposition = seasonal_decompose(ts_data.fillna(ts_data.mean()), model='additive', period=12)
        df['trend_component'] = df['date'].map(
            lambda x: decomposition.trend[decomposition.trend.index == x.strftime('%Y-%m')].values[0] 
            if x.strftime('%Y-%m') in decomposition.trend.index else np.nan)
    except:
        df['trend_component'] = df['score'].mean()
    return df

def prepare_data(df):
    clientes_validos = df['customer_id'].value_counts()[df['customer_id'].value_counts() >= 2].index
    df_model = df[df['customer_id'].isin(clientes_validos)].copy()
    df_model['next_score'] = df_model.groupby('customer_id')['score'].shift(-1)
    df_model['next_date'] = df_model.groupby('customer_id')['date'].shift(-1)
    df_model['days_until_next'] = (df_model['next_date'] - df_model['date']).dt.days
    if 'secondary_driver' in df_model.columns:
        df_model['next_secondary_driver'] = df_model.groupby('customer_id')['secondary_driver'].shift(-1)
    df_model = df_model.dropna(subset=['next_score', 'next_date', 'days_until_next'])
    return df_model

def train_models(df_model):
    features_score = [
        'month_sin', 'month_cos', 'quarter', 'year',
        'days_since_last_survey', 'survey_frequency',
        'score_ma_3', 'score_ma_6', 'score_std_3',
        'historical_promoter_rate', 'trend_component',
        'primary_driver_freq', 'primary_driver_target',
        'secondary_driver_freq', 'secondary_driver_target'
    ]
    features_days = [
        'month_sin', 'month_cos', 'quarter', 'year',
        'days_since_last_survey', 'survey_frequency',
        'score_ma_3', 'historical_promoter_rate',
        'primary_driver_freq', 'secondary_driver_freq'
    ]
    features_driver = [
        'primary_driver_freq', 'primary_driver_target',
        'month_sin', 'month_cos', 'quarter', 'year',
        'score_ma_3', 'historical_promoter_rate'
    ]
    categorical_features = ['quarter']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    pipeline_score = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ])
    # XGBoost NO usa min_samples_leaf, así que lo quitamos
    param_grid_score = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10, 20]
    }
    tscv = TimeSeriesSplit(n_splits=3)
    grid_score = GridSearchCV(pipeline_score, param_grid_score, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_score.fit(df_model[features_score], df_model['next_score'])
    model_score = grid_score.best_estimator_
    pipeline_days = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(random_state=42))
    ])
    grid_days = GridSearchCV(pipeline_days, param_grid_score, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_days.fit(df_model[features_days], df_model['days_until_next'])
    model_days = grid_days.best_estimator_
    model_driver = None
    if 'next_secondary_driver' in df_model.columns and len(df_model['next_secondary_driver'].unique()) > 1:
        le_driver = LabelEncoder()
        y_driver = le_driver.fit_transform(df_model['next_secondary_driver'].fillna('Unknown'))
        joblib.dump(le_driver, 'secondary_driver_label_encoder.pkl')
        pipeline_driver = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])
        param_grid_driver = {
            'classifier__n_estimators': [100, 150],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_leaf': [1, 5]
        }
        grid_driver = GridSearchCV(pipeline_driver, param_grid_driver, cv=tscv, scoring='accuracy', n_jobs=-1)
        grid_driver.fit(df_model[features_driver], y_driver)
        model_driver = grid_driver.best_estimator_
    return model_score, model_days, model_driver

def generate_predictions(df, model_score, model_days, model_driver=None):
    df_pred = df.sort_values(['customer_id', 'date']).groupby('customer_id').last().reset_index()
    if 'days_until_next' in df.columns:
        historical_days = df['days_until_next'].dropna()
        p10 = historical_days.quantile(0.1)
        p90 = historical_days.quantile(0.9)
    else:
        p10, p90 = 30, 365
    prediction_months = pd.date_range(start=PREDICTION_START, end=PREDICTION_END, freq='MS')
    all_predictions = []
    driver_encoder = None
    if model_driver and os.path.exists('secondary_driver_label_encoder.pkl'):
        driver_encoder = joblib.load('secondary_driver_label_encoder.pkl')
    for pred_date in prediction_months:
        temp_df = df_pred.copy()
        temp_df['prediction_month'] = pred_date
        temp_df['days_since_last'] = (pred_date - temp_df['date']).dt.days
        temp_df['month'] = pred_date.month
        temp_df['year'] = pred_date.year
        temp_df['quarter'] = (pred_date.month - 1) // 3 + 1
        temp_df['month_sin'] = np.sin(2 * np.pi * temp_df['month']/12)
        temp_df['month_cos'] = np.cos(2 * np.pi * temp_df['month']/12)
        features_score = [
            'month_sin', 'month_cos', 'quarter', 'year',
            'days_since_last_survey', 'survey_frequency',
            'score_ma_3', 'score_ma_6', 'score_std_3',
            'historical_promoter_rate', 'trend_component',
            'primary_driver_freq', 'primary_driver_target',
            'secondary_driver_freq', 'secondary_driver_target'
        ]
        features_days = [
            'month_sin', 'month_cos', 'quarter', 'year',
            'days_since_last_survey', 'survey_frequency',
            'score_ma_3', 'historical_promoter_rate',
            'primary_driver_freq', 'secondary_driver_freq'
        ]
        temp_df['predicted_score'] = model_score.predict(temp_df[features_score])
        temp_df['predicted_days'] = model_days.predict(temp_df[features_days])
        temp_df['days_until_pred_month'] = (pred_date - temp_df['date']).dt.days
        temp_df['predicted_days'] = temp_df['predicted_days'].clip(lower=p10, upper=p90)
        temp_df['next_survey_date'] = temp_df['date'] + pd.to_timedelta(temp_df['predicted_days'], unit='d')
        mask = (
            (temp_df['next_survey_date'].dt.to_period('M') == pred_date.to_period('M')) |
            ((temp_df['next_survey_date'] - pred_date).abs().dt.days <= 15)
        )
        monthly_pred = temp_df[mask].copy()
        if not monthly_pred.empty and model_driver and 'primary_driver_freq' in monthly_pred.columns:
            features_driver = [
                'primary_driver_freq', 'primary_driver_target',
                'month_sin', 'month_cos', 'quarter', 'year',
                'score_ma_3', 'historical_promoter_rate'
            ]
            driver_pred = model_driver.predict(monthly_pred[features_driver])
            monthly_pred['predicted_secondary_driver'] = driver_encoder.inverse_transform(driver_pred)
        if not monthly_pred.empty:
            all_predictions.append(monthly_pred)
    if not all_predictions:
        raise ValueError("No se generaron predicciones para el período especificado")
    df_pred = pd.concat(all_predictions)
    df_pred['predicted_category'] = pd.cut(df_pred['predicted_score'],
                                         bins=[-1, 6, 8, 11],
                                         labels=['Detractor', 'Passive', 'Promoter'])
    df_pred = df_pred.sort_values(['customer_id', 'prediction_month'])
    df_pred = df_pred.drop_duplicates(['customer_id', 'prediction_month'])
    return df_pred

def predict_next_survey_dates(df):
    print("\n🔎 Calculando ciclo real de encuestas por cliente (ajustado)...")
    df = df.sort_values(['customer_id', 'date'])
    df['days_between'] = df.groupby('customer_id')['date'].diff().dt.days

    # Mediana global para respaldo
    global_median = df['days_between'].median()

    # Ciclo ajustado: moda si std < 5 y hay al menos 3 valores, si no mediana, si no global
    def ciclo_mas_probable(x):
        x = x.dropna()
        if len(x) >= 3 and x.std() < 5:
            return x.mode().iloc[0] if not x.mode().empty else x.median()
        elif len(x) >= 1:
            return x.median()
        else:
            return global_median

    ciclo_cliente = df.groupby('customer_id')['days_between'].agg(ciclo_mas_probable)
    ultima_fecha = df.groupby('customer_id')['date'].max()
    # Día de la semana típico
    df['weekday'] = df['date'].dt.weekday
    weekday_cliente = df.groupby('customer_id')['weekday'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 0)

    # Tomar el último valor real de cada cliente para esas columnas
    extra_cols = []
    for col in ['ddc_name', 'pdv', 'region', 'sub_region', 'segment']:
        if col in df.columns:
            extra_cols.append(col)
    last_real = df.sort_values(['customer_id', 'date']).groupby('customer_id').last()

    # Agregar la categoría real del último score ANTES de crear extra_data
    if 'score' in last_real.columns:
        last_real['category'] = pd.cut(
            last_real['score'],
            bins=[-1, 6, 8, 11],
            labels=['Detractor', 'Passive', 'Promoter']
        )

    extra_data = last_real[extra_cols + (['category'] if 'category' in last_real.columns else [])] if extra_cols or 'category' in last_real.columns else None

    predicciones = pd.DataFrame({
        'customer_id': ultima_fecha.index,
        'last_survey_date': ultima_fecha.values,
        'cycle_days': ciclo_cliente.values,
        'weekday': weekday_cliente.values
    })
    predicciones['next_survey_date'] = predicciones['last_survey_date'] + pd.to_timedelta(predicciones['cycle_days'], unit='d')
    # Ajustar al día típico
    def ajusta_al_dia_semana(fecha, dia_objetivo):
        if pd.isnull(fecha):
            return fecha
        fecha = pd.Timestamp(fecha)
        while fecha.weekday() != dia_objetivo:
            fecha += pd.Timedelta(days=1)
        return fecha
    predicciones['next_survey_date'] = [
        ajusta_al_dia_semana(row['next_survey_date'], row['weekday'])
        for _, row in predicciones.iterrows()
    ]
    # Añadir columnas reales y categoría
    if extra_data is not None:
        predicciones = predicciones.merge(extra_data, left_on='customer_id', right_index=True, how='left')
    print("\n✅ Predicción basada en ciclo real ajustada completada.")
    return predicciones

def generate_reports(df, df_pred):
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df = df.dropna(subset=['score'])
    try:
        plt.figure(figsize=(12, 6))
        trend_data = df.groupby(df['date'].dt.to_period('M'))['score'].mean().reset_index()
        trend_data['date'] = trend_data['date'].astype(str)
        sns.lineplot(data=trend_data, x='date', y='score')
        plt.title('Tendencia histórica de NPS')
        plt.xlabel('Fecha')
        plt.ylabel('Score promedio')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('historical_trend.png')
        plt.close()
    except Exception as e:
        print(f"❌ Error al generar gráfico de tendencia histórica: {str(e)}")
    try:
        plt.figure(figsize=(8, 5))
        df_pred['predicted_category'].value_counts().plot(kind='bar')
        plt.title('Distribución de categorías NPS predichas')
        plt.tight_layout()
        plt.savefig('predicted_categories.png')
        plt.close()
    except Exception as e:
        print(f"❌ Error al generar gráfico de categorías predichas: {str(e)}")
    if 'predicted_secondary_driver' in df_pred.columns:
        try:
            plt.figure(figsize=(12, 6))
            df_pred['predicted_secondary_driver'].value_counts().head(10).plot(kind='bar')
            plt.title('Top 10 drivers secundarios predichos')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('top_drivers.png')
            plt.close()
            driver_score = df_pred.groupby('predicted_secondary_driver')['predicted_score'].mean().sort_values(ascending=False)
            driver_score.to_excel('driver_score_relationship.xlsx')
        except Exception as e:
            print(f"❌ Error al generar reporte de drivers secundarios: {str(e)}")
    try:
        report_data = {
            'total_clientes': df['customer_id'].nunique(),
            'total_predicciones': len(df_pred),
            'score_promedio_predicho': df_pred['predicted_score'].mean(),
            'proporcion_promoters': (df_pred['predicted_category'] == 'Promoter').mean(),
            'proporcion_detractores': (df_pred['predicted_category'] == 'Detractor').mean()
        }
        pd.DataFrame.from_dict(report_data, orient='index').to_excel('reporte_consolidado.xlsx')
    except Exception as e:
        print(f"❌ Error al generar reporte consolidado: {str(e)}")
    dashboard_data = {
        'historical_metrics': {
            'monthly_trends': df.groupby(df['date'].dt.to_period('M'))['score'].mean().reset_index(),
            'category_dist': df['category'].value_counts(normalize=True).mul(100).round(1) if 'category' in df.columns else None,
            'top_drivers': df['primary_driver'].value_counts().nlargest(10) if 'primary_driver' in df.columns else None
        },
        'prediction_months': df_pred['prediction_month'].unique().tolist()
    }
    joblib.dump(dashboard_data, 'nps_dashboard_data.pkl')

def main():
    print("\n" + "="*50)
    print("🚀 INICIANDO SISTEMA DE PREDICCIÓN NPS")
    print("="*50 + "\n")
    print("🔍 Cargando datos...")
    df = load_data("Historico clientes Nps.xlsx")
    if df is None:
        print("❌ No se pudo cargar el archivo. Verifica el formato y los datos.")
        return
    print("\n🔧 Creando características avanzadas...")
    df = create_features(df)
    print("\n📊 Preparando datos para el modelo...")
    try:
        df_model = prepare_data(df)
    except ValueError as e:
        print(f"❌ Error: {str(e)}")
        return
    print("\n🤖 Entrenando modelos...")
    model_score, model_days, model_driver = train_models(df_model)
    print("\n💾 Guardando modelos entrenados...")
    joblib.dump(model_score, 'nps_score_predictor.pkl')
    joblib.dump(model_days, 'nps_days_predictor.pkl')
    if model_driver:
        joblib.dump(model_driver, 'nps_driver_predictor.pkl')
    print("\n🔮 Generando predicciones futuras...")
    try:
        df_pred = generate_predictions(df, model_score, model_days, model_driver)
    except Exception as e:
        print(f"❌ Error al generar predicciones: {str(e)}")
        return
    # Exportar predicciones completas (modelo)
    cols_to_export = [
        'customer_id', 'ddc_name', 'pdv', 'region', 'sub_region', 'segment', 
        'score', 'category', 'date', 'primary_driver', 'secondary_driver',
        'prediction_month', 'predicted_score', 'predicted_category',
        'predicted_days', 'next_survey_date'
    ]
    if 'predicted_secondary_driver' in df_pred.columns:
        cols_to_export.append('predicted_secondary_driver')
    cols_to_export = [c for c in cols_to_export if c in df_pred.columns]
    df_pred[cols_to_export].to_excel('predicciones_nps_completas.xlsx', index=False)
    # Exportar predicción por ciclo real ajustado (ahora con columnas reales y categoría)
    ciclo_pred = predict_next_survey_dates(df)
    ciclo_pred.to_excel('prediccion_ciclo_real.xlsx', index=False)
    generate_reports(df, df_pred)
    print("\n" + "="*50)
    print("✅ PROCESO COMPLETADO CON ÉXITO")
    print("="*50)
    print("\n📌 Archivos generados:")
    print("- predicciones_nps_completas.xlsx: Predicciones detalladas (modelo)")
    print("- prediccion_ciclo_real.xlsx: Predicción basada en ciclo real ajustado")
    print("- nps_score_predictor.pkl: Modelo para predicción de scores")
    print("- nps_days_predictor.pkl: Modelo para predicción de frecuencia")
    if model_driver:
        print("- nps_driver_predictor.pkl: Modelo para predicción de drivers")
    print("\n🎯 ¡Listo para analizar los resultados!")

if __name__ == "__main__":
    main()