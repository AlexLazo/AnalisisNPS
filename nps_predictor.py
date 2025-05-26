import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, StackingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
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
from prophet import Prophet
import optuna
from sklearn.base import clone

warnings.filterwarnings('ignore')

# Configuración global
PREDICTION_START = datetime(2025, 1, 1)
PREDICTION_END = datetime(2026, 12, 31)

# 1. Funciones mejoradas de carga y preprocesamiento
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
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            df = df.dropna(subset=['score'])
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

def handle_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.05)
        q3 = df[col].quantile(0.95)
        iqr = q3 - q1
        df[col] = np.where(df[col] > q3 + 1.5*iqr, q3, 
                          np.where(df[col] < q1 - 1.5*iqr, q1, df[col]))
    return df

def add_prophet_features(df):
    try:
        prophet_data = df.groupby('date')['score'].mean().reset_index()
        prophet_data.columns = ['ds', 'y']
        
        m = Prophet(seasonality_mode='multiplicative', yearly_seasonality=True)
        m.fit(prophet_data)
        
        future = m.make_future_dataframe(periods=0)
        forecast = m.predict(future)
        
        df = df.merge(forecast[['ds', 'trend', 'yearly']], 
                     left_on='date', right_on='ds', how='left')
        df = df.rename(columns={
            'trend': 'prophet_trend',
            'yearly': 'prophet_seasonality'
        }).drop('ds', axis=1)
    except Exception as e:
        print(f"⚠️ Error al agregar características de Prophet: {str(e)}")
        df['prophet_trend'] = df['score'].mean()
        df['prophet_seasonality'] = 0
    
    return df

# 2. Ingeniería de características mejorada
def enhanced_feature_engineering(df):
    # Verificar columnas mínimas requeridas
    required_columns = ['customer_id', 'date', 'score']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Columna requerida '{col}' no encontrada en los datos")

    # Hacer copia para evitar SettingWithCopyWarning
    df = df.copy()
    
    # Ordenar datos
    df = df.sort_values(['customer_id', 'date'])
    
    try:
        # Features temporales básicas
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['quarter'] = df['date'].dt.quarter
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Features de secuencia temporal
        df['days_since_last_survey'] = df.groupby('customer_id')['date'].diff().dt.days.fillna(0)
        df['days_since_first_survey'] = (df['date'] - df.groupby('customer_id')['date'].transform('min')).dt.days.fillna(0)
        df['survey_count'] = df.groupby('customer_id').cumcount() + 1
        df['survey_frequency'] = df.groupby('customer_id')['date'].transform('count') / 365
        
        # Moving averages
        for window in [3, 6, 12]:
            df[f'score_ma_{window}'] = df.groupby('customer_id')['score'].transform(
                lambda x: x.rolling(window, min_periods=1).mean())
        
        # Tendencia
        def safe_trend_calc(x):
            try:
                if len(x) >= 2:
                    return np.polyfit(range(len(x)), x, 1)[0]
                return 0
            except:
                return 0
        
        df['score_trend'] = df.groupby('customer_id')['score'].transform(
            lambda x: pd.Series(x).rolling(3, min_periods=1).apply(safe_trend_calc))
        
        # Interacciones
        df['score_times_freq'] = df['score'] * df['survey_frequency']
        
        # Categorías históricas
        if 'category' in df.columns:
            df['historical_promoter_rate'] = df.groupby('customer_id')['category'].transform(
                lambda x: (x == 'Promoter').expanding().mean().fillna(0))
        else:
            df['historical_promoter_rate'] = 0
        
        # Drivers
        for driver_col in ['primary_driver', 'secondary_driver']:
            if driver_col in df.columns:
                df[f'{driver_col}_freq'] = df[driver_col].map(
                    df[driver_col].value_counts(normalize=True))
                df[f'{driver_col}_target_mean'] = df.groupby(driver_col)['score'].transform('mean')
                df[f'{driver_col}_target_median'] = df.groupby(driver_col)['score'].transform('median')
        
        # Componentes temporales
        try:
            ts_data = df.groupby('date')['score'].mean()
            decomposition = seasonal_decompose(ts_data, model='additive', period=12)
            df['trend_component'] = df['date'].map(decomposition.trend)
        except:
            df['trend_component'] = df['score'].mean()
        
        # Prophet features
        df = add_prophet_features(df)
        
    except Exception as e:
        print(f"⚠️ Error en ingeniería de características: {str(e)}")
        # Asegurar que todas las columnas necesarias existan
        default_columns = {
            'month': df['date'].dt.month,
            'year': df['date'].dt.year,
            'quarter': df['date'].dt.quarter,
            'month_sin': 0,
            'month_cos': 0,
            'days_since_last_survey': 0,
            'survey_count': 1,
            'survey_frequency': 1/365,
            'score_ma_3': df['score'],
            'score_trend': 0,
            'historical_promoter_rate': 0,
            'trend_component': df['score'].mean(),
            'prophet_trend': df['score'].mean(),
            'prophet_seasonality': 0,
            'score_times_freq': df['score']
        }
        
        for col, default_value in default_columns.items():
            if col not in df.columns:
                df[col] = default_value
    
    return df

# 3. Preparación de datos
def prepare_data(df):
    clientes_validos = df['customer_id'].value_counts()[df['customer_id'].value_counts() >= 2].index
    df_model = df[df['customer_id'].isin(clientes_validos)].copy()
    
    df_model['next_score'] = df_model.groupby('customer_id')['score'].shift(-1)
    df_model['next_date'] = df_model.groupby('customer_id')['date'].shift(-1)
    df_model['days_until_next'] = (df_model['next_date'] - df_model['date']).dt.days
    
    if 'secondary_driver' in df_model.columns:
        df_model['next_secondary_driver'] = df_model.groupby('customer_id')['secondary_driver'].shift(-1)
    
    # Columnas requeridas mínimas
    required_cols = ['next_score', 'next_date', 'days_until_next']
    df_model = df_model.dropna(subset=[col for col in required_cols if col in df_model.columns])
    
    return df_model

# 4. Optimización de hiperparámetros con Optuna
def optimize_hyperparameters(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        
        model = XGBRegressor(**params, random_state=42)
        scores = -cross_val_score(
            model, X, y, 
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_absolute_error'
        )
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, n_jobs=-1)
    
    return study.best_params

# 5. Modelado mejorado
def build_enhanced_models(df_model):
    # Verificar columnas disponibles
    available_cols = df_model.columns.tolist()
    print("\nColumnas disponibles en df_model:", available_cols)
    
    # Definición de features dinámicas
    def get_available_features(feature_list):
        return [col for col in feature_list if col in available_cols]
    
    # Features base para score
    base_features_score = [
        'month_sin', 'month_cos', 'quarter', 'year',
        'days_since_last_survey', 'survey_frequency', 'survey_count',
        'score_trend', 'historical_promoter_rate',
        'trend_component', 'prophet_trend', 'prophet_seasonality',
        'primary_driver_target_mean', 'primary_driver_target_median',
        'secondary_driver_target_mean', 'secondary_driver_target_median',
        'score_times_freq'
    ]
    
    # Features base para días
    base_features_days = [
        'month_sin', 'month_cos', 'quarter', 'year',
        'days_since_last_survey', 'survey_frequency', 'survey_count',
        'score_ma_3', 'historical_promoter_rate',
        'primary_driver_freq', 'secondary_driver_freq',
        'prophet_trend'
    ]
    
    # Obtener features disponibles
    features_score = get_available_features(base_features_score)
    features_days = get_available_features(base_features_days)
    
    print("\nFeatures para modelo de score:", features_score)
    print("Features para modelo de días:", features_days)
    
    if not features_score or not features_days:
        raise ValueError("No hay suficientes características disponibles para entrenar los modelos")
    
    # Preprocesamiento para score
    preprocessor_score = ColumnTransformer([
        ('num', 'passthrough', features_score)
    ])
    
    # Preprocesamiento para días
    preprocessor_days = ColumnTransformer([
        ('num', 'passthrough', features_days)
    ])
    
    # Optimización para el modelo de score
    print("\n🔍 Optimizando modelo de score...")
    X_score = df_model[features_score]
    y_score = df_model['next_score']
    score_params = optimize_hyperparameters(X_score, y_score)
    
    # Modelo final de score
    model_score = Pipeline([
        ('preprocessor', preprocessor_score),
        ('regressor', XGBRegressor(**score_params, random_state=42))
    ])
    model_score.fit(X_score, y_score)
    
    # Modelo para días
    print("\n🔍 Entrenando modelo de días...")
    model_days = Pipeline([
        ('preprocessor', preprocessor_days),
        ('regressor', XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ))
    ])
    
    if 'days_until_next' not in df_model.columns:
        print("⚠️ Advertencia: 'days_until_next' no está disponible, saltando entrenamiento del modelo de días")
        model_days = None
    else:
        model_days.fit(df_model[features_days], df_model['days_until_next'])
    
    return model_score, model_days, None

# 6. Validación mejorada
def enhanced_validation(model, X, y, model_name=""):
    """
    Validates a model using time series cross-validation and generates performance metrics and plots.
    """
    # Create time series splits
    tscv = TimeSeriesSplit(n_splits=3)
    predictions = []
    actuals = []
    
    # Perform time series cross-validation
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Clone the model to avoid fitting the same model multiple times
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)
        
        predictions.extend(y_pred)
        actuals.extend(y_test)
    
    # Convert to numpy arrays for metric calculation
    y_pred = np.array(predictions)
    y_true = np.array(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n📊 Métricas de Validación para {model_name}:")
    print(f"- MAE: {mae:.3f}")
    print(f"- RMSE: {rmse:.3f}")
    print(f"- R²: {r2:.3f}")
    
    # Gráfico de residuales
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f'Distribución de Residuales - {model_name}')
    plt.xlabel('Error de Predicción')
    plt.show()
    
    # Gráfico de valores reales vs predichos
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.title(f'Valores Reales vs Predichos - {model_name}')
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.show()

# 7. Generación de predicciones (compatible con dashboard)
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
        
        # Features actualizadas dinámicamente
        available_cols = temp_df.columns.tolist()
        
        def get_features(feature_list):
            return [col for col in feature_list if col in available_cols]
        
        features_score = get_features([
            'month_sin', 'month_cos', 'quarter', 'year',
            'days_since_last_survey', 'survey_frequency', 'survey_count',
            'score_std_3', 'score_trend', 'historical_promoter_rate',
            'trend_component', 'prophet_trend', 'prophet_seasonality',
            'primary_driver_target_mean', 'primary_driver_target_median',
            'secondary_driver_target_mean', 'secondary_driver_target_median',
            'score_times_freq'
        ])
        
        features_days = get_features([
            'month_sin', 'month_cos', 'quarter', 'year',
            'days_since_last_survey', 'survey_frequency', 'survey_count',
            'score_ma_3', 'historical_promoter_rate',
            'primary_driver_freq', 'secondary_driver_freq',
            'prophet_trend'
        ])
        
        # Predicciones solo si tenemos las características necesarias
        if all(col in temp_df.columns for col in features_score):
            temp_df['predicted_score'] = model_score.predict(temp_df[features_score])
        
        if all(col in temp_df.columns for col in features_days):
            temp_df['predicted_days'] = model_days.predict(temp_df[features_days])
            temp_df['predicted_days'] = temp_df['predicted_days'].clip(lower=p10, upper=p90)
            temp_df['next_survey_date'] = temp_df['date'] + pd.to_timedelta(temp_df['predicted_days'], unit='d')
        
        mask = (
            (temp_df['next_survey_date'].dt.to_period('M') == pred_date.to_period('M')) |
            ((temp_df['next_survey_date'] - pred_date).abs().dt.days <= 15)
        ) if 'next_survey_date' in temp_df.columns else pd.Series(False, index=temp_df.index)
        
        monthly_pred = temp_df[mask].copy()
        
        if not monthly_pred.empty and model_driver:
            features_driver = get_features([
                'primary_driver_freq', 'primary_driver_target_mean',
                'month_sin', 'month_cos', 'quarter', 'year',
                'score_ma_3', 'historical_promoter_rate',
                'prophet_seasonality'
            ])
            if all(col in monthly_pred.columns for col in features_driver):
                driver_pred = model_driver.predict(monthly_pred[features_driver])
                monthly_pred['predicted_secondary_driver'] = driver_encoder.inverse_transform(driver_pred)
        
        if not monthly_pred.empty:
            all_predictions.append(monthly_pred)
    
    if not all_predictions:
        raise ValueError("No se generaron predicciones para el período especificado")
    
    df_pred = pd.concat(all_predictions)
    if 'predicted_score' in df_pred.columns:
        df_pred['predicted_category'] = pd.cut(df_pred['predicted_score'],
                                             bins=[-1, 6, 8, 11],
                                             labels=['Detractor', 'Passive', 'Promoter'])
    df_pred = df_pred.sort_values(['customer_id', 'prediction_month'])
    df_pred = df_pred.drop_duplicates(['customer_id', 'prediction_month'])
    
    return df_pred

# 8. Funciones de reportes (compatibles con dashboard existente)
def generate_reports(df, df_pred):
    # Gráfico de tendencia histórica
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
    
    # Gráfico de categorías predichas
    try:
        plt.figure(figsize=(8, 5))
        if 'predicted_category' in df_pred.columns:
            df_pred['predicted_category'].value_counts().plot(kind='bar')
            plt.title('Distribución de categorías NPS predichas')
            plt.tight_layout()
            plt.savefig('predicted_categories.png')
            plt.close()
    except Exception as e:
        print(f"❌ Error al generar gráfico de categorías predichas: {str(e)}")
    
    # Reporte de drivers (si existe)
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
    
    # Reporte consolidado (compatible con dashboard)
    try:
        report_data = {
            'total_clientes': df['customer_id'].nunique(),
            'total_predicciones': len(df_pred),
            'score_promedio_predicho': df_pred['predicted_score'].mean() if 'predicted_score' in df_pred.columns else None,
            'proporcion_promoters': (df_pred['predicted_category'] == 'Promoter').mean() if 'predicted_category' in df_pred.columns else None,
            'proporcion_detractores': (df_pred['predicted_category'] == 'Detractor').mean() if 'predicted_category' in df_pred.columns else None
        }
        pd.DataFrame.from_dict(report_data, orient='index').to_excel('reporte_consolidado.xlsx')
    except Exception as e:
        print(f"❌ Error al generar reporte consolidado: {str(e)}")
    
    # Datos para dashboard (formato compatible)
    dashboard_data = {
        'historical_metrics': {
            'monthly_trends': df.groupby(df['date'].dt.to_period('M'))['score'].mean().reset_index(),
            'category_dist': df['category'].value_counts(normalize=True).mul(100).round(1) if 'category' in df.columns else None,
            'top_drivers': df['primary_driver'].value_counts().nlargest(10) if 'primary_driver' in df.columns else None
        },
        'prediction_months': df_pred['prediction_month'].unique().tolist() if 'prediction_month' in df_pred.columns else []
    }
    joblib.dump(dashboard_data, 'nps_dashboard_data.pkl')

# 9. Función principal
def main():
    print("\n" + "="*50)
    print("🚀 INICIANDO SISTEMA DE PREDICCIÓN NPS MEJORADO")
    print("="*50 + "\n")
    
    # Carga de datos
    print("🔍 Cargando y preparando datos...")
    df = load_data("Historico clientes Nps.xlsx")
    if df is None:
        return
    
    # Ingeniería de características
    print("\n🔧 Realizando ingeniería de características avanzada...")
    df = enhanced_feature_engineering(df)
    
    # Preparación de datos para modelado
    print("\n📊 Preparando datos para modelado...")
    try:
        df_model = prepare_data(df)
        if len(df_model) < 100:
            raise ValueError("Datos insuficientes para modelado (necesitas al menos 100 muestras)")
    except ValueError as e:
        print(f"❌ Error: {str(e)}")
        return
    
    # Entrenamiento de modelos
    print("\n🤖 Entrenando modelos avanzados...")
    model_score, model_days, model_driver = build_enhanced_models(df_model)
    
    # Validación
    print("\n🧪 Validando modelos...")
    features_score = [col for col in [
        'month_sin', 'month_cos', 'quarter', 'year',
        'days_since_last_survey', 'survey_frequency', 'survey_count',
        'score_std_3', 'score_trend', 'historical_promoter_rate',
        'trend_component', 'prophet_trend', 'prophet_seasonality',
        'primary_driver_target_mean', 'primary_driver_target_median',
        'secondary_driver_target_mean', 'secondary_driver_target_median',
        'score_times_freq'
    ] if col in df_model.columns]
    
    enhanced_validation(model_score, df_model[features_score], df_model['next_score'], "Modelo de Score")
    
    # Guardado de modelos (compatible con dashboard)
    print("\n💾 Guardando modelos...")
    joblib.dump(model_score, 'nps_score_predictor.pkl')
    joblib.dump(model_days, 'nps_days_predictor.pkl')
    if model_driver:
        joblib.dump(model_driver, 'nps_driver_predictor.pkl')
    
    # Generación de predicciones
    print("\n🔮 Generando predicciones futuras...")
    try:
        print("\n🔮 Generando predicciones futuras...")
        df_pred = generate_predictions(df, model_score, model_days, model_driver)
        print(f"Predicciones generadas: {len(df_pred)} registros")
        
        print("\n📤 Exportando resultados...")
        print("Columnas disponibles:", df_pred.columns.tolist())
        
    except Exception as e:
        print(f"❌ Error detallado: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return
    
    # Exportación de resultados
    print("\n📤 Exportando resultados...")
    cols_to_export = [
        'customer_id', 'ddc_name', 'pdv', 'region', 'sub_region', 'segment', 
        'score', 'category', 'date', 'primary_driver', 'secondary_driver',
        'prediction_month', 'predicted_score', 'predicted_category',
        'predicted_days', 'next_survey_date'
    ]
    cols_to_export = [c for c in cols_to_export if c in df_pred.columns]
    df_pred[cols_to_export].to_excel('predicciones_nps_completas.xlsx', index=False)
    
    # Generación de reportes (compatibles con dashboard)
    generate_reports(df, df_pred)
    
    print("\n" + "="*50)
    print("✅ PROCESO COMPLETADO CON ÉXITO")
    print("="*50)
    print("\n📌 Archivos generados:")
    print("- predicciones_nps_completas.xlsx: Predicciones detalladas")
    print("- historical_trend.png: Tendencia histórica")
    print("- predicted_categories.png: Distribución de categorías predichas")
    print("- reporte_consolidado.xlsx: Métricas consolidadas")
    print("- nps_dashboard_data.pkl: Datos para dashboard (formato compatible)")
    print("\n🎯 ¡Listo para analizar los resultados!")

if __name__ == "__main__":
    main()