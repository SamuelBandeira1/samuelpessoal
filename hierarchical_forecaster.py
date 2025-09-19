"""Pipeline completo de previsão hierárquica com segmentação estatística.

Este módulo reorganiza o script original para facilitar a leitura sem alterar
nenhuma instrução existente. Cada etapa está comentada para explicitar seu
papel no processo de modelagem e previsão.

FORECAST (ABC + Segmentação Estatística + MixedLM + Croston + SARIMA + HW + WLS)
======================================================================================

Este script une o melhor das versões anteriores:
  • Pré-processamento robusto + validações
  • ABC (Volume/Valor/Frequência) com score ponderado
  • Classificação estatística (estacionariedade, intermitência, tendência, sazonalidade, volatilidade)
  • Segmentação hierárquica → modelos por segmento:
      - Intermittent → Croston
      - Seasonal → Auto-SARIMA enxuto
      - Trend → Regressão polinomial
      - Volatile → Holt‑Winters (AIC)
      - Stable/Other → MixedLM ponderado (Yeo‑Johnson + dummies mensais)
      - Fallback geral → Média móvel exponencialmente ponderada
  • Transformações por segmento (Box‑Cox/log1p/none) + inversão segura
  • Recalibração WLS + variance smoothing (LOWESS)
  • Validação (R², MAPE, RMSE/MAE, Shapiro, Breusch‑Pagan, CV temporal)
  • Forecast futuro hierárquico (UF→Linha→Grupo→Produto) via YOY + Rolling com limites
  • Exportação para Excel (chunking) + gráficos

Requer colunas: ['UF','CD_HIER_N11','Venda Vol','Venda P.M','Mes','Ano'].
Para o forecast hierárquico, usa ['DS_LINHA','DS_GRUPO'] (se não existirem, pula só essa parte).
"""

# =============================================================================
# 1. IMPORTAÇÕES
# =============================================================================
import re
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import TimeSeriesSplit

from scipy.stats import boxcox, shapiro
from scipy.special import inv_boxcox
from scipy import stats

from dateutil.relativedelta import relativedelta

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 2. CONSTANTES GERAIS E CONFIGURAÇÕES
# =============================================================================

def _sanitize_filename(s: str) -> str:
    """Normaliza nomes de arquivos substituindo caracteres inválidos."""
    s = str(s)
    s = s.strip().replace(' ', '_')
    return re.sub(r'[:*?"<>|/\\]+', '_', s)


FILE_PATH = Path(r"C:\\Users\\samuel.bandeira\\Desktop\\Base_Estatistica_TCC.xlsx")
OUTPUT_PATH = Path(r"C:\\Users\\samuel.bandeira\\Desktop\\resultado_forecast_6m.xlsx")
_OUTPUT_BASE = OUTPUT_PATH.parent
CHARTS_DIR = (_OUTPUT_BASE / "charts").resolve()
CHARTS_LINES_DIR = (_OUTPUT_BASE / "charts_lines").resolve()

HORIZONTE = 6
CUTOFF    = "2024-12-31"
MIN_HISTORICO_MESES = 24

ALPHA_CROSTON = 0.1
BETA_CROSTON  = 0.1
CROSTON_TUNE = True
CROSTON_ALPHA_GRID = (0.05, 0.1, 0.2, 0.3)
CROSTON_BETA_GRID  = (0.05, 0.1, 0.2, 0.3)
INTERMITTENT_ZERO_RATIO = 0.40

FAST_MODE = True
REPORT_WAPE = True

SEASONAL_CV_THRESHOLD = 0.20
SEASONAL_ACF12_MIN    = 0.30
VOLATILE_CV_THRESHOLD = 0.8

PESO_YOY      = 0.01
WINDOW_BOUND  = 6
FATOR_LIMITE  = 1.1

CAP_P95_MULT       = 1.15
CAP_MAX_MULT       = 1.15
CAP_RECENT_WINDOW  = 2
CAP_RECENT_MULT    = 1.50
CAP_FUT_MULT       = 1.05

EXCEL_MAX_ROWS = 1_048_576

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# 3. FUNÇÕES UTILITÁRIAS
# =============================================================================

def wape(y_true, y_pred, sample_weight=None) -> float:
    """Weighted Absolute Percentage Error (também conhecido como WMAPE/WAPE)."""
    import numpy as np
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp) & (yt >= 0)
    if not mask.any():
        return float("nan")
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)[mask]
        w = np.where(np.isfinite(w), w, 0.0)
        num = np.sum(np.abs(yt[mask] - yp[mask]) * w)
        den = np.sum(yt[mask] * w)
    else:
        num = float(np.sum(np.abs(yt[mask] - yp[mask])))
        den = float(np.sum(yt[mask]))
    return float(num / den * 100.0) if den > 0 else float("inf")


def mape(y_true: Union[List[float], np.ndarray], y_pred: Union[List[float], np.ndarray]) -> float:
    """Calcula o erro percentual absoluto médio (MAPE)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float('inf')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o DataFrame em conjuntos de treino e teste a partir do corte temporal."""
    cut = pd.to_datetime(CUTOFF)
    train = df[df['Data'] <= cut].copy()
    test  = df[df['Data'] >  cut].copy()
    return train, test


def safe_inverse_boxcox(value: float, lam: float) -> float:
    """Aplica inversa de Box-Cox de forma segura."""
    try:
        return float(inv_boxcox(value, lam))
    except Exception:
        return float(value)


# =============================================================================
# 4. CLASSE PRINCIPAL DO PIPELINE
# =============================================================================

class Forecaster:
    """Classe que encapsula todo o fluxo de preparação, modelagem e exportação."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.target_transformers: Dict[str, Tuple[str, Any]] = {}
        self.pt_yj: PowerTransformer = PowerTransformer(method='yeo-johnson')
        self.dummy_cols: List[str] = []
        self.segment_metrics: Dict[str, Any] = {}
        self.exog_base_cols: list[str] = ['Ano_Linear', 'Dias_Uteis']
        self.exog_cols: list[str] = []

    # ------------------------------------------------------------------
    # 4.1 CARREGAMENTO E PRÉ-PROCESSAMENTO
    # ------------------------------------------------------------------
    def load_and_preprocess(self) -> pd.DataFrame:
        """Carrega a base, cria variáveis auxiliares e aplica transformações iniciais."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.file_path}")

        df = pd.read_excel(self.file_path)

        df = df.rename(columns={'Venda P.M': 'Venda_PM', 'Venda Vol': 'Venda_Vol', 'Mês': 'Mes'})
        req = ['UF','CD_HIER_N11','Venda_Vol','Venda_PM','Ano']
        for col in req:
            if col not in df.columns:
                raise ValueError(f"Coluna obrigatória ausente: {col}")

        if 'Data' not in df.columns:
            if 'Mes' not in df.columns:
                raise ValueError("Precisa de 'Mes' ou 'Data' na planilha.")
            df['Data'] = pd.to_datetime(df['Ano'].astype(str) + '-' + df['Mes'].astype(str).str.zfill(2))
        else:
            df['Data'] = pd.to_datetime(df['Data'])

        df['UF'] = df['UF'].astype('category')
        df['CD_HIER_N11'] = df['CD_HIER_N11'].astype('category')

        df = df[df['Venda_Vol'] >= 0].copy()

        df['Mes_Num'] = df['Data'].dt.month
        df['Trimestre'] = df['Data'].dt.quarter
        df['Ano_Linear'] = df['Ano'] - df['Ano'].min()

        monthly = (
            df.groupby(['UF','CD_HIER_N11','Mes_Num'], observed=True)['Venda_Vol']
              .mean()
              .unstack(fill_value=0)
        )
        cv = monthly.std(axis=1) / (monthly.mean(axis=1).replace(0, np.nan))
        cv = cv.replace([np.inf, -np.inf], np.nan).fillna(0)
        saz_df = cv.rename('cv').reset_index()
        saz_df['is_seasonal_cv'] = saz_df['cv'] >= SEASONAL_CV_THRESHOLD
        df = df.merge(saz_df[['UF','CD_HIER_N11','is_seasonal_cv']], on=['UF','CD_HIER_N11'], how='left')

        df = pd.get_dummies(df, columns=['Mes_Num'], prefix='Mes', drop_first=True)
        self.dummy_cols = [c for c in df.columns if c.startswith('Mes_')]

        if 'Dias_Uteis' not in df.columns:
            df['_AnoMes'] = df['Data'].dt.to_period('M')
            dias_map = {}
            for per in df['_AnoMes'].unique():
                rng = pd.date_range(start=per.start_time, end=per.end_time, freq='D')
                dias_map[per] = int((rng.dayofweek < 5).sum())
            df['Dias_Uteis'] = df['_AnoMes'].map(dias_map).astype(int)
            df.drop(columns=['_AnoMes'], inplace=True)
        self.exog_cols = [c for c in (self.exog_base_cols) if c in df.columns]

        vol_mean = df['Venda_Vol'].mean()
        vol_mad  = (df['Venda_Vol'] - df['Venda_Vol'].median()).abs().median()
        denom = max(1.0, vol_mean if np.isfinite(vol_mean) and vol_mean>0 else 1.0)
        df['weights'] = (df['Venda_Vol'].clip(lower=0)) / denom
        df['weights'] = df['weights'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        df['sqrt_w'] = np.sqrt(df['weights'])

        df['transf_venda'] = self.pt_yj.fit_transform(df[['Venda_Vol']])
        df['transf_venda_tilde'] = df['transf_venda'] * df['sqrt_w']
        df['Venda_PM_tilde'] = df['Venda_PM'] * df['sqrt_w']
        df['Ano_tilde'] = df['Ano'] * df['sqrt_w']
        for c in self.dummy_cols:
            df[c + '_tilde'] = df[c] * df['sqrt_w']

        logger.info(f"Dados: {len(df):,} linhas de {df['Data'].min().date()} a {df['Data'].max().date()}")
        return df

    # ------------------------------------------------------------------
    # 4.2 CLASSIFICAÇÕES ESTATÍSTICAS
    # ------------------------------------------------------------------
    def classify_abc_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica classificação ABC multidimensional (volume, valor e frequência)."""
        agg = df.groupby('CD_HIER_N11').agg(
            Venda_Vol_sum=('Venda_Vol','sum'),
            Venda_Vol_std=('Venda_Vol','std'),
            cont=('Venda_Vol','count'),
            Venda_PM_mean=('Venda_PM','mean'),
            Data_nunique=('Data','nunique')
        ).reset_index()

        agg['Valor_Total'] = agg['Venda_Vol_sum'] * agg['Venda_PM_mean']
        agg['CV_Volume'] = agg['Venda_Vol_std'] / agg['Venda_Vol_sum'].replace(0, np.nan)
        agg['CV_Volume'] = agg['CV_Volume'].replace([np.inf, -np.inf], np.nan).fillna(0)
        agg['Frequencia'] = agg['Data_nunique']

        def abc_from_cum(series: pd.Series) -> pd.Categorical:
            s = series.sort_values(ascending=False)
            perc = s.cumsum() / s.sum()
            bins = pd.cut(perc, bins=[0, 0.80, 0.95, 1.0], labels=['A','B','C'], include_lowest=True)
            return bins.reindex(series.sort_values(ascending=False).index).reindex(series.index)

        agg['ABC_Volume'] = abc_from_cum(agg.set_index('CD_HIER_N11')['Venda_Vol_sum']).values
        agg['ABC_Valor']  = abc_from_cum(agg.set_index('CD_HIER_N11')['Valor_Total']).values

        freq_sorted = agg.sort_values('Frequencia', ascending=False)
        freq_perc = freq_sorted['Frequencia'].cumsum()/max(agg['Frequencia'].sum(), 1)
        freq_cat = pd.cut(freq_perc, bins=[0,0.80,0.95,1.0], labels=['A','B','C'], include_lowest=True)
        agg['ABC_Frequencia'] = freq_cat.values

        scores = {'A': 3, 'B': 2, 'C': 1}
        def combine(row):
            a = row.get('ABC_Volume', 'C') or 'C'
            v = row.get('ABC_Valor', 'C') or 'C'
            f = row.get('ABC_Frequencia', 'C') or 'C'
            total = scores.get(a, 1)*0.5 + scores.get(v, 1)*0.3 + scores.get(f, 1)*0.2
            if total >= 2.5:   return 'A'
            elif total >= 1.8: return 'B'
            else:              return 'C'

        agg['ABC_Final'] = agg.apply(combine, axis=1)
        return agg[['CD_HIER_N11','ABC_Final','CV_Volume','Frequencia']]

    def classify_time_series_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identifica estacionariedade, intermitência, tendência e sazonalidade das séries."""
        expected_cols = [
            'UF','CD_HIER_N11','is_stationary','is_intermittent','has_trend',
            'trend_direction','is_seasonal','is_volatile','cv','zero_ratio','avg_interval'
        ]
        rows = []
        for (uf, prod), g in df.groupby(['UF','CD_HIER_N11']):
            if g['Data'].nunique() < MIN_HISTORICO_MESES:
                continue
            ts = g.sort_values('Data').set_index('Data')['Venda_Vol']

            try:
                _, adf_p, *_ = adfuller(ts.dropna())
                is_stationary = adf_p < 0.05
            except Exception:
                is_stationary = False

            zero_ratio = (ts == 0).mean()
            nz_idx = np.where(ts.values > 0)[0]
            avg_interval = float(np.diff(nz_idx).mean()) if len(nz_idx) > 1 else np.inf
            is_intermittent = (zero_ratio > INTERMITTENT_ZERO_RATIO) or (avg_interval > 2)

            has_trend, trend_z = self._mann_kendall(ts)
            trend_direction = 'increasing' if trend_z > 0 else ('decreasing' if trend_z < 0 else 'none')

            monthly = ts.groupby(ts.index.month).mean()
            monthly_cv = (monthly.std() / monthly.mean()) if monthly.mean() > 0 else 0

            acf_sig = False
            if monthly_cv <= SEASONAL_CV_THRESHOLD:
                vals = ts.fillna(0.0)
                n_eff = max(10, len(vals))
                try:
                    acf_vals = acf(vals, nlags=12, fft=True)
                    acf12 = float(acf_vals[12]) if len(acf_vals) >= 13 else 0.0
                    acf_thresh = 1.96 / np.sqrt(n_eff)
                    acf_sig = abs(acf12) > acf_thresh
                except Exception:
                    acf_sig = False
            is_seasonal = (monthly_cv > SEASONAL_CV_THRESHOLD) or acf_sig

            cv = (ts.std() / ts.mean()) if ts.mean() > 0 else 0
            is_volatile = cv > VOLATILE_CV_THRESHOLD

            rows.append({
                'UF': uf,
                'CD_HIER_N11': prod,
                'is_stationary': is_stationary,
                'is_intermittent': is_intermittent,
                'has_trend': has_trend,
                'trend_direction': trend_direction,
                'is_seasonal': is_seasonal,
                'is_volatile': is_volatile,
                'cv': float(cv),
                'zero_ratio': float(zero_ratio),
                'avg_interval': float(0 if np.isinf(avg_interval) else avg_interval),
            })

        if not rows:
            logger.warning("classify_time_series_advanced: sem séries >= MIN_HISTORICO_MESES; retornando placeholder vazio.")
            return pd.DataFrame(columns=expected_cols)
        return pd.DataFrame(rows)

    @staticmethod
    def _mann_kendall(x: pd.Series) -> Tuple[bool, float]:
        """Executa o teste de Mann-Kendall para identificar tendências."""
        x = x.dropna()
        n = len(x)
        if n < 3:
            return False, 0.0
        s = 0
        for k in range(n-1):
            s += np.sign(x.iloc[k+1:] - x.iloc[k]).sum()
        var_s = (n*(n-1)*(2*n+5))/18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0.0
        p = 2*(1 - stats.norm.cdf(abs(z)))
        return p < 0.10, float(z)

    # ------------------------------------------------------------------
    # 4.3 SEGMENTAÇÃO E TRANSFORMAÇÕES
    # ------------------------------------------------------------------
    def create_segments_hierarchical(self, df: pd.DataFrame, abc_df: pd.DataFrame, ts_df: pd.DataFrame) -> pd.DataFrame:
        """Combina informações ABC e estatísticas para definir segmentos de modelagem."""
        df = df.merge(abc_df, on='CD_HIER_N11', how='left')
        df = df.merge(ts_df, on=['UF','CD_HIER_N11'], how='left')

        def seg(row):
            abc = row.get('ABC_Final', 'C')
            if row.get('is_intermittent', False):
                return f"{abc}_Intermittent"
            if abc in ['A','B'] and row.get('is_volatile', False):
                return f"{abc}_Volatile"
            if abc in ['A','B'] and row.get('is_seasonal', False) and row.get('has_trend', False):
                return f"{abc}_Seasonal_Trend"
            if abc in ['A','B'] and row.get('is_seasonal', False):
                return f"{abc}_Seasonal"
            if abc in ['A','B'] and row.get('has_trend', False):
                return f"{abc}_Trend"
            if not any([row.get('is_intermittent', False), row.get('is_volatile', False), row.get('is_seasonal', False), row.get('has_trend', False)]):
                return f"{abc}_Stable"
            return f"{abc}_Other"

        df['segment'] = df.apply(seg, axis=1)
        return df

    def apply_target_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica transformações Box-Cox/log1p por segmento para estabilizar variância."""
        df = df.copy()
        for seg in df['segment'].dropna().unique():
            mask = df['segment'] == seg
            y = df.loc[mask, 'Venda_Vol'].values.astype(float)
            y_shift = y + 1.0
            try:
                orig_p = shapiro(np.clip(y_shift, 0, None))[1] if len(y_shift) >= 3 else 0
                y_bc, lam = boxcox(y_shift)
                bc_p = shapiro(y_bc[:min(5000, len(y_bc))])[1] if len(y_bc) >= 3 else 0
                if bc_p > orig_p * 1.2:
                    df.loc[mask, 'Venda_Vol_Trans'] = y_bc
                    self.target_transformers[seg] = ('boxcox', lam)
                else:
                    df.loc[mask, 'Venda_Vol_Trans'] = np.log1p(y)
                    self.target_transformers[seg] = ('log1p', None)
            except Exception:
                df.loc[mask, 'Venda_Vol_Trans'] = y
                self.target_transformers[seg] = ('none', None)
        return df

    # ------------------------------------------------------------------
    # 4.4 MODELAGEM POR SEGMENTO
    # ------------------------------------------------------------------
    @staticmethod
    def croston_forecast(ts: pd.Series, alpha: float = ALPHA_CROSTON, beta: float = BETA_CROSTON) -> float:
        """Executa Croston SBA para séries intermitentes."""
        x = np.asarray(ts, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return 0.0

        demands, intervals = [], []
        last_idx = None
        for i, v in enumerate(x):
            if v > 0:
                demands.append(v)
                if last_idx is not None:
                    intervals.append(i - last_idx)
                last_idx = i

        if len(demands) == 0:
            return 0.0
        if len(demands) == 1:
            return float(np.mean(x[x > 0]))

        d_hat = demands[0]
        for d in demands[1:]:
            d_hat = alpha * d + (1 - alpha) * d_hat

        p_hat = intervals[0] if len(intervals) > 0 else 1.0
        for p in intervals[1:]:
            p_hat = beta * p + (1 - beta) * p_hat

        croston_rate = d_hat / max(p_hat, 1e-9)
        sba = (1 - alpha / 2.0) * croston_rate
        return float(max(0.0, sba))

    @staticmethod
    def _inverse_transform_value(val: float, t: Tuple[str, Any]) -> float:
        """Reverte a transformação aplicada ao alvo."""
        kind, param = t
        try:
            if kind == 'boxcox':
                return max(0.0, inv_boxcox(val, param) - 1.0)
            if kind == 'log1p':
                return max(0.0, np.expm1(val))
            return max(0.0, float(val))
        except Exception:
            return max(0.0, float(val))

    def auto_arima_seasonal(self, ts: pd.Series, max_p: int = 2, max_q: int = 2):
        """Busca simples de SARIMA sazonal baseada em AIC."""
        ts = pd.Series(ts).astype(float)
        if ts.dropna().size < 18 or float(ts.dropna().std()) < 1e-6:
            return None, {}
        best_aic = float('inf'); best_model = None; best_params = {}
        for p in range(max_p+1):
            for d in [0,1]:
                for q in range(max_q+1):
                    for P in [0,1]:
                        for D in [0,1]:
                            for Q in [0,1]:
                                try:
                                    m = SARIMAX(ts, order=(p,d,q),
                                                seasonal_order=(P,D,Q,12),
                                                enforce_stationarity=False, enforce_invertibility=False)
                                    r = m.fit(disp=False, maxiter=100)
                                    if r.aic < best_aic:
                                        best_aic = r.aic; best_model = r
                                        best_params = {'order':(p,d,q), 'seasonal_order':(P,D,Q,12), 'aic': float(r.aic)}
                                except Exception:
                                    continue
        return best_model, best_params

    def auto_sarimax_cv(self, ts: pd.Series, exog: pd.DataFrame | None, max_p: int = 2, max_q: int = 2):
        """Seleciona SARIMAX via validação cruzada temporal."""
        ts = pd.Series(ts).astype(float)
        if ts.dropna().size < 18 or float(ts.dropna().std()) < 1e-6:
            return None, {}

        if exog is not None:
            exog = exog.reindex(ts.index).astype(float)

        def _mape(y_true, y_pred):
            y_true = pd.Series(y_true).astype(float)
            y_pred = pd.Series(y_pred).astype(float)
            mask = (y_true > 0) & y_true.notna() & y_pred.notna()
            if not mask.any(): return float('inf')
            return float((abs((y_true[mask]-y_pred[mask])/y_true[mask])).mean()*100)

        grid = []
        for p in range(max_p+1):
            for d in [0,1]:
                for q in range(max_q+1):
                    for P in [0,1]:
                        for D in [0,1]:
                            for Q in [0,1]:
                                grid.append(((p,d,q),(P,D,Q,12)))

        tscv = TimeSeriesSplit(n_splits=3)
        best = {'mape': float('inf'), 'order': None, 'seasonal_order': None, 'aic': float('inf')}
        best_fit = None

        for (order, sorder) in grid:
            try:
                fold_mapes = []
                for tr_idx, va_idx in tscv.split(ts):
                    y_tr = ts.iloc[tr_idx]; y_va = ts.iloc[va_idx]
                    ex_tr = exog.iloc[tr_idx] if exog is not None else None
                    ex_va = exog.iloc[va_idx] if exog is not None else None

                    mdl = SARIMAX(y_tr, order=order, seasonal_order=sorder,
                                  exog=ex_tr, enforce_stationarity=False, enforce_invertibility=False)
                    res = mdl.fit(disp=False, maxiter=200)

                    pred = res.get_forecast(steps=len(y_va), exog=ex_va).predicted_mean
                    fold_mapes.append(_mape(y_va, pred))

                mean_mape = float(pd.Series(fold_mapes).mean())
                mdl_final = SARIMAX(ts, order=order, seasonal_order=sorder, exog=exog,
                                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)
                aic_val = float(mdl_final.aic)

                if (mean_mape < best['mape']) or (mean_mape == best['mape'] and aic_val < best['aic']):
                    best = {'mape': mean_mape, 'order': order, 'seasonal_order': sorder, 'aic': aic_val}
                    best_fit = mdl_final
            except Exception:
                continue

        if best_fit is None:
            return None, {}
        best['cv_mape'] = best.pop('mape')
        return best_fit, best

    def fit_models_with_fallback(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Executa modelagem por segmento com mecanismos de fallback."""
        train_res = []
        test_res  = []
        seg_metrics: Dict[str, Any] = {}

        for seg in train['segment'].dropna().unique():
            tr = train[train['segment']==seg].copy()
            te = test[test['segment']==seg].copy()
            if tr.empty:
                continue
            logger.info(f"Modelando segmento: {seg} (train={len(tr)})")
            try:
                if 'Intermittent' in seg:
                    tri, tei, mets = self._fit_intermittent(tr, te)
                elif 'Seasonal' in seg:
                    tri, tei, mets = self._fit_seasonal(tr, te)
                elif 'Trend' in seg:
                    tri, tei, mets = self._fit_trend(tr, te)
                elif 'Volatile' in seg:
                    tri, tei, mets = self._fit_volatile(tr, te)
                else:
                    if 'Stable' in seg:
                        tri, tei, mets = self._fit_stable_mixedlm(tr, te)
                    elif 'Other' in seg:
                        tri, tei, mets = self._fit_fallback(tr, te)
                    else:
                        pass
            except Exception as e:
                logger.warning(f"Falha no segmento {seg} → fallback. Detalhe: {e}")
                tri, tei, mets = self._fit_fallback(tr, te)
            train_res.append(tri)
            test_res.append(tei)
            seg_metrics[seg] = mets

        train_out = pd.concat(train_res, ignore_index=True) if train_res else pd.DataFrame()
        test_out  = pd.concat(test_res,  ignore_index=True) if test_res  else pd.DataFrame()
        return train_out, test_out, seg_metrics

    def _fit_intermittent(self, train: pd.DataFrame, test: pd.DataFrame):
        """Modelagem para séries intermitentes utilizando Croston com tuning leve."""
        tri = train.copy()
        tei = test.copy()
        tri['yhat'] = np.nan
        tei['yhat'] = np.nan

        def _croston_one(ts_orig: pd.Series) -> float:
            x = ts_orig.sort_index()
            base_fc = self.croston_forecast(x, alpha=ALPHA_CROSTON, beta=BETA_CROSTON)

            if not CROSTON_TUNE or len(x) < 18:
                return float(max(0.0, base_fc))

            split_k = min(6, max(3, len(x)//4))
            y_tr = x.iloc[:-split_k]
            y_va = x.iloc[-split_k:]

            best = (float("inf"), ALPHA_CROSTON, BETA_CROSTON, base_fc)
            for a in CROSTON_ALPHA_GRID:
                for b in CROSTON_BETA_GRID:
                    try:
                        fc_rate = self.croston_forecast(y_tr, alpha=a, beta=b)
                        pred = np.repeat(fc_rate, len(y_va))
                        score = wape(y_va.values, pred)
                        if score < best[0]:
                            best = (score, a, b, fc_rate)
                    except Exception:
                        continue
            return float(max(0.0, best[3]))

        for (uf, prod), g in train.groupby(['UF','CD_HIER_N11']):
            g = g.sort_values('Data')
            ts_orig = g.set_index('Data')['Venda_Vol']
            f = _croston_one(ts_orig)
            inv = float(max(0.0, f))

            tri.loc[g.index, 'yhat'] = inv
            gtest = test[(test['UF']==uf) & (test['CD_HIER_N11']==prod)]
            if not gtest.empty:
                tei.loc[gtest.index, 'yhat'] = inv

        mets = self._basic_metrics(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0), model='Croston')
        return tri, tei, mets

    def _fit_seasonal(self, train: pd.DataFrame, test: pd.DataFrame):
        """Modelagem para séries sazonais usando Holt-Winters em modo rápido."""
        tri = train.copy(); tei = test.copy()
        tri['yhat'] = np.nan; tei['yhat'] = np.nan

        if FAST_MODE:
            cfgs = [
                dict(trend='add', seasonal='add', seasonal_periods=12),
                dict(trend='add', seasonal=None),
                dict(trend=None, seasonal='add', seasonal_periods=12),
                dict(trend=None, seasonal=None),
            ]
            for (uf, prod), g in train.groupby(['UF','CD_HIER_N11']):
                g = g.sort_values('Data')
                ts = g.set_index('Data')['Venda_Vol_Trans']
                seg_name = g['segment'].iloc[0]

                best_fit = None; best_aic = np.inf
                for c in cfgs:
                    try:
                        mdl = ExponentialSmoothing(ts, **c)
                        res = mdl.fit(optimized=True, remove_bias=True, use_brute=False)
                        if res.aic < best_aic:
                            best_fit, best_aic = res, res.aic
                    except Exception:
                        continue

                if best_fit is None:
                    mean_val = float(ts.mean())
                    inv = self._inverse_transform_value(mean_val, self.target_transformers[seg_name])
                    tri.loc[g.index, 'yhat'] = inv
                    tg = test[(test['UF']==uf)&(test['CD_HIER_N11']==prod)]
                    if not tg.empty: tei.loc[tg.index, 'yhat'] = inv
                else:
                    tri.loc[g.index, 'yhat'] = [
                        self._inverse_transform_value(v, self.target_transformers[seg_name])
                        for v in best_fit.fittedvalues
                    ]
                    tg = test[(test['UF']==uf)&(test['CD_HIER_N11']==prod)].sort_values('Data')
                    if not tg.empty:
                        fc = best_fit.forecast(steps=len(tg))
                        tei.loc[tg.index, 'yhat'] = [
                            self._inverse_transform_value(v, self.target_transformers[seg_name]) for v in fc
                        ]

            mets = self._basic_metrics(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0),
                                       model='ETS_fast', extra={})
            return tri, tei, mets

    def _fit_trend(self, train: pd.DataFrame, test: pd.DataFrame):
        """Modelagem de séries com tendência via regressão polinomial."""
        tri = train.copy(); tei = test.copy()
        tri['yhat'] = np.nan; tei['yhat'] = np.nan
        r2_scores = []

        for (uf, produto), g in train.groupby(['UF','CD_HIER_N11']):
            g = g.sort_values('Data')
            ts = g.set_index('Data')['Venda_Vol_Trans']
            seg_name = g['segment'].iloc[0]
            n = len(ts)
            if n < 3:
                mean_val = float(ts.mean())
                mean_orig = self._inverse_transform_value(mean_val, self.target_transformers[seg_name])
                tri.loc[g.index, 'yhat'] = mean_orig
                tg = test[(test['UF']==uf) & (test['CD_HIER_N11']==produto)]
                if not tg.empty:
                    tei.loc[tg.index, 'yhat'] = mean_orig
                r2_scores.append(0.0)
                continue

            time_idx = np.arange(n)
            best_model = None; best_X = None; best_r2 = -np.inf

            max_degree = min(3, max(1, n-1))
            for degree in range(1, max_degree+1):
                try:
                    X = np.column_stack([time_idx**i for i in range(1, degree+1)])
                    X = sm.add_constant(X, has_constant='add')
                    model = sm.OLS(ts, X).fit()
                    if model.rsquared > best_r2:
                        best_r2 = model.rsquared; best_model = model; best_X = X
                except Exception:
                    continue

            if best_model is None:
                mean_val = float(ts.mean())
                mean_orig = self._inverse_transform_value(mean_val, self.target_transformers[seg_name])
                tri.loc[g.index, 'yhat'] = mean_orig
                tg = test[(test['UF']==uf) & (test['CD_HIER_N11']==produto)]
                if not tg.empty:
                    tei.loc[tg.index, 'yhat'] = mean_orig
                r2_scores.append(0.0)
                continue

            train_pred = best_model.predict(best_X)
            train_pred_orig = [self._inverse_transform_value(v, self.target_transformers[seg_name]) for v in train_pred]
            tri.loc[g.index, 'yhat'] = np.array(train_pred_orig, dtype=float)

            tg = test[(test['UF']==uf) & (test['CD_HIER_N11']==produto)].sort_values('Data')
            if not tg.empty:
                steps = len(tg)
                test_time_idx = np.arange(n, n+steps)
                n_poly = int(len(best_model.params) - 1)
                if n_poly <= 0:
                    X_test = np.ones((steps, 1))
                else:
                    X_test = np.column_stack([test_time_idx**i for i in range(1, n_poly+1)])
                    X_test = sm.add_constant(X_test, has_constant='add')

                p = len(best_model.params)
                if X_test.shape[1] != p:
                    if p-1 <= 0:
                        X_test = np.ones((steps, 1))
                    else:
                        X_test = np.column_stack([test_time_idx**i for i in range(1, (p-1)+1)])
                        X_test = sm.add_constant(X_test, has_constant='add')

                try:
                    test_pred = best_model.predict(X_test)
                except Exception:
                    test_pred = np.full(steps, train_pred.iloc[-1])

                test_pred_orig = [self._inverse_transform_value(v, self.target_transformers[seg_name]) for v in test_pred]
                train_cap = np.percentile(train_pred_orig, 95) if len(train_pred_orig) else None
                if train_cap is not None and np.isfinite(train_cap):
                    cap = 1.2 * max(1e-9, train_cap)
                    test_pred_orig = np.clip(test_pred_orig, 0.0, cap)

                tei.loc[tg.index, 'yhat'] = np.array(test_pred_orig, dtype=float)

            r2_scores.append(best_r2)

        mets = {
            'model_type': 'Polynomial_Trend',
            'r2': r2_score(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0)) if len(tei) else 0.0,
            'mape': mape(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0)) if len(tei) else np.inf,
            'rmse': float(np.sqrt(mean_squared_error(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0)))) if len(tei) else np.nan,
            'mae': mean_absolute_error(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0)) if len(tei) else np.nan,
            'avg_insample_r2': float(np.mean(r2_scores)) if r2_scores else 0.0,
            'n_obs': int(len(tei))
        }
        return tri, tei, mets

    def _fit_volatile(self, train: pd.DataFrame, test: pd.DataFrame):
        """Modelagem para séries voláteis utilizando Holt-Winters com seleção de AIC."""
        tri = train.copy(); tei = test.copy()
        tri['yhat'] = np.nan; tei['yhat'] = np.nan
        aics=[]

        for (uf, prod), g in train.groupby(['UF','CD_HIER_N11']):
            g = g.sort_values('Data')
            ts = g.set_index('Data')['Venda_Vol_Trans']
            seg_name = g['segment'].iloc[0]
            best = None; best_aic = np.inf
            configs = [
                {'trend':'add','seasonal':None},
                {'trend':'add','seasonal':'add','seasonal_periods':12},
                {'trend':'mul','seasonal':None},
                {'trend':None,'seasonal':None}
            ]
            for c in configs:
                try:
                    m = ExponentialSmoothing(ts, trend=c.get('trend'), seasonal=c.get('seasonal'),
                                             seasonal_periods=c.get('seasonal_periods',12))
                    r = m.fit(optimized=True, remove_bias=True)
                    if r.aic < best_aic:
                        best, best_aic = r, r.aic
                except Exception:
                    continue

            if best is None:
                alpha=0.3
                smoothed=[ts.iloc[0]] if len(ts)>0 else [0.0]
                for i in range(1,len(ts)):
                    smoothed.append(alpha*ts.iloc[i] + (1-alpha)*smoothed[-1])
                last = smoothed[-1]
                inv_last = self._inverse_transform_value(float(last), self.target_transformers[seg_name])
                tri.loc[g.index, 'yhat'] = inv_last
                gtest = test[(test['UF']==uf)&(test['CD_HIER_N11']==prod)]
                if not gtest.empty:
                    tei.loc[gtest.index, 'yhat'] = inv_last
                aics.append(np.inf)
            else:
                trp = best.fittedvalues
                tri.loc[g.index, 'yhat'] = [self._inverse_transform_value(v, self.target_transformers[seg_name]) for v in trp]
                gtest = test[(test['UF']==uf)&(test['CD_HIER_N11']==prod)]
                if not gtest.empty:
                    fc = best.forecast(steps=len(gtest))
                    tei.loc[gtest.index, 'yhat'] = [self._inverse_transform_value(v, self.target_transformers[seg_name]) for v in fc]
                aics.append(best_aic)

        extra = {'aic_avg': float(np.mean(aics)) if aics else np.inf}
        mets = self._basic_metrics(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0), model='Holt_Winters', extra=extra)
        return tri, tei, mets

    def _fit_stable_mixedlm(self, train: pd.DataFrame, test: pd.DataFrame):
        """Modelagem para séries estáveis via MixedLM ponderado."""
        tri = train.copy(); tei=test.copy()
        tri['yhat'] = np.nan; tei['yhat'] = np.nan

        f_seasonal    = "transf_venda_tilde ~ Venda_PM_tilde + Ano_tilde + " + " + ".join([c+"_tilde" for c in self.dummy_cols]) if self.dummy_cols else "transf_venda_tilde ~ Venda_PM_tilde + Ano_tilde"
        f_nonseasonal = "transf_venda_tilde ~ Venda_PM_tilde + Ano_tilde"

        def fit_predict(df_in: pd.DataFrame, formula: str) -> pd.Series:
            if df_in.empty:
                return pd.Series([], dtype=float)
            m = sm.MixedLM.from_formula(
                formula=formula,
                groups=df_in['UF'],
                re_formula='1',
                vc_formula={
                    'Produto': '0 + C(CD_HIER_N11)',
                    'Prod_P' : '0 + C(CD_HIER_N11):Venda_PM_tilde'
                },
                data=df_in
            ).fit(method='lbfgs', disp=False)
            y_tilde = m.predict(df_in)
            y_transf = y_tilde / df_in['sqrt_w'].replace(0, np.nan).fillna(1.0)
            y_transf = np.array(y_transf).reshape(-1,1)
            inv = self.pt_yj.inverse_transform(y_transf).flatten()
            inv = np.clip(inv, 0, None)
            return pd.Series(inv, index=df_in.index)

        tr_s = tri[tri['is_seasonal_cv']==True]
        tr_n = tri[tri['is_seasonal_cv']!=True]
        te_s = tei[tei['is_seasonal_cv']==True]
        te_n = tei[tei['is_seasonal_cv']!=True]

        try:
            tri.loc[tr_s.index, 'yhat'] = fit_predict(tr_s, f_seasonal)
        except Exception:
            tri.loc[tr_s.index, 'yhat'] = fit_predict(tr_s, f_nonseasonal)
        try:
            tri.loc[tr_n.index, 'yhat'] = fit_predict(tr_n, f_nonseasonal)
        except Exception:
            tri.loc[tr_n.index, 'yhat'] = fit_predict(tr_n, f_seasonal)
        try:
            tei.loc[te_s.index, 'yhat'] = fit_predict(te_s, f_seasonal)
        except Exception:
            tei.loc[te_s.index, 'yhat'] = fit_predict(te_s, f_nonseasonal)
        try:
            tei.loc[te_n.index, 'yhat'] = fit_predict(te_n, f_nonseasonal)
        except Exception:
            tei.loc[te_n.index, 'yhat'] = fit_predict(te_n, f_seasonal)

        mets = self._basic_metrics(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0), model='MixedLM', extra={})
        return tri, tei, mets

    def _fit_fallback(self, train: pd.DataFrame, test: pd.DataFrame):
        """Fallback baseado em média móvel exponencial ponderada."""
        tri = train.copy(); tei=test.copy()
        tri['yhat'] = np.nan; tei['yhat'] = np.nan

        for (uf, prod), g in train.groupby(['UF','CD_HIER_N11']):
            ts = g.sort_values('Data').set_index('Data')['Venda_Vol']
            if len(ts) >= 6:
                weights = np.exp(np.linspace(0,1,min(6,len(ts))))
                recent = ts.tail(len(weights))
                wavg = float(np.average(recent, weights=weights))
            else:
                wavg = float(ts.mean())
            tri.loc[g.index, 'yhat'] = wavg
            gtest = test[(test['UF']==uf)&(test['CD_HIER_N11']==prod)]
            if not gtest.empty:
                tei.loc[gtest.index, 'yhat'] = wavg

        mets = self._basic_metrics(tei['Venda_Vol'].fillna(0), tei['yhat'].fillna(0), model='Weighted_Avg_Fallback', extra={})
        return tri, tei, mets

    @staticmethod
    def _basic_metrics(y_true, y_pred, model: str, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calcula conjunto padrão de métricas de avaliação."""
        extra = extra or {}
        try:
            return {
                'model_type': model,
                'r2': r2_score(y_true, y_pred),
                'mape': mape(y_true, y_pred),
                'wape': wape(y_true, y_pred),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': mean_absolute_error(y_true, y_pred),
                **extra
            }
        except Exception:
            return {'model_type': model, **extra}

    # ------------------------------------------------------------------
    # 4.5 RECALIBRAÇÃO, CAPPING E VALIDAÇÃO
    # ------------------------------------------------------------------
    @staticmethod
    def apply_wls_smoothing(hist: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Recalibração por WLS com suavização de variância."""
        df = hist.copy()
        df['resid_orig'] = (df['Venda_Vol'] - df['yhat']).astype(float)
        df['var_resid'] = (df['resid_orig']**2).replace(0, np.nan)
        min_pos = df['var_resid'].dropna().min() if df['var_resid'].dropna().size>0 else 1e-9
        df['var_resid'] = df['var_resid'].fillna(min_pos)
        df['w_puro'] = 1.0 / df['var_resid']

        X = sm.add_constant(df[['yhat']].astype(float))
        try:
            wls_puro = sm.WLS(df['Venda_Vol'].astype(float), X, weights=df['w_puro']).fit()
            df['yhat_wls'] = wls_puro.predict(X)
        except Exception:
            df['yhat_wls'] = df['yhat']

        df['resid_wls'] = df['Venda_Vol'] - df['yhat_wls']

        try:
            smoothed = lowess(endog=df['var_resid'], exog=df['yhat_wls'], frac=0.3, return_sorted=True)
            df['var_smooth'] = np.interp(df['yhat_wls'], smoothed[:,0], smoothed[:,1])
        except Exception:
            df['var_smooth'] = df['var_resid']

        min_vs = df['var_smooth'].dropna().min() if df['var_smooth'].dropna().size>0 else 1e-9
        df['var_smooth'] = df['var_smooth'].replace(0, min_vs).fillna(min_vs)
        df['w_smooth'] = 1.0 / df['var_smooth']

        try:
            wls_smooth = sm.WLS(df['Venda_Vol'].astype(float), X, weights=df['w_smooth']).fit()
            df['yhat_wls_smooth'] = wls_smooth.predict(X)
        except Exception:
            df['yhat_wls_smooth'] = df['yhat_wls']

        try:
            lm_stat_o, lm_p_o, _, _ = het_breuschpagan(df['resid_orig'], X)
            lm_stat_s, lm_p_s, _, _ = het_breuschpagan(df['Venda_Vol'] - df['yhat_wls_smooth'], X)
            diag = {'bp_orig_p': float(lm_p_o), 'bp_wls_smooth_p': float(lm_p_s)}
        except Exception:
            diag = {'bp_orig_p': 0.0, 'bp_wls_smooth_p': 0.0}

        return df, diag

    def compute_caps_per_series(self, train: pd.DataFrame) -> pd.DataFrame:
        """Calcula limites superiores plausíveis por série com base no histórico."""
        caps: List[Dict[str, Any]] = []
        for (uf, prod), g in train.groupby(['UF', 'CD_HIER_N11']):
            g = g.sort_values('Data')
            real = g['Venda_Vol'].astype(float).dropna()
            if real.empty:
                continue

            p95 = float(np.percentile(real, 95)) if real.size >= 3 else float(real.max())
            mx  = float(real.max())
            recent = float(real.tail(min(CAP_RECENT_WINDOW, len(real))).mean())

            base = max(p95 * CAP_P95_MULT, recent * CAP_RECENT_MULT)
            if mx > 0:
                upper = min(base, mx * CAP_MAX_MULT)
            else:
                upper = base

            if (not np.isfinite(upper)) or (upper <= 0):
                upper = max(float(real.mean()) * CAP_RECENT_MULT, 1.0)

            caps.append({'UF': uf, 'CD_HIER_N11': prod, 'cap_upper': float(upper)})

        return pd.DataFrame(caps)

    def validate(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Avalia o desempenho global, por segmento e por linha."""
        y_true_all = hist['Venda_Vol']
        y_pred_all = hist['yhat_final']

        mask = y_true_all.notna() & y_pred_all.notna()
        y_true = y_true_all[mask]
        y_pred = y_pred_all[mask]
        coverage = float(mask.mean())

        results: Dict[str, Any] = {
            'global_metrics': {
                'r2_global'   : r2_score(y_true, y_pred) if len(y_true) else np.nan,
                'mape_global' : mape(y_true, y_pred)     if len(y_true) else np.nan,
                'wmape_global': wape(y_true, y_pred)     if len(y_true) else np.nan,
                'rmse_global' : float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan,
                'mae_global'  : mean_absolute_error(y_true, y_pred) if len(y_true) else np.nan,
                'n_observations': int(len(y_true)),
                'coverage_rate' : coverage
            }
        }

        if 'DS_LINHA' in hist.columns:
            linha_mets = []
            for ln, g in hist.groupby('DS_LINHA'):
                if len(g) < 10:
                    continue
                linha_mets.append({
                    'linha': ln,
                    'mape': mape(g['Venda_Vol'], g['yhat_final']),
                    'wape': wape(g['Venda_Vol'], g['yhat_final']),
                    'r2': r2_score(g['Venda_Vol'], g['yhat_final']),
                    'rmse': float(np.sqrt(mean_squared_error(g['Venda_Vol'], g['yhat_final']))),
                    'mae': mean_absolute_error(g['Venda_Vol'], g['yhat_final']),
                    'n_obs': int(len(g))
                })
            results['linha_metrics'] = linha_mets

            try:
                dfm = pd.DataFrame(linha_mets)
                mape_global_agg = float(np.average(dfm['mape'], weights=dfm['n_obs']))
                results['global_metrics']['mape_global'] = mape_global_agg
                results['global_metrics']['wmape_global'] = wape(y_true, y_pred)
                gm = results['global_metrics']
                logger.info(
                    f"R² Global: {gm['r2_global']:.4f} | MAPE Global: {gm['mape_global']:.2f}% | WAPE Global: {gm['wmape_global']:.2f}%"
                )
            except Exception:
                pass

        try:
            stat, p = shapiro((y_true - y_pred)[:min(5000,len(hist))])
            results['residual_normality'] = {'shapiro_stat': float(stat), 'shapiro_p': float(p), 'is_normal': p>0.05}
        except Exception as e:
            results['residual_normality'] = {'error': str(e)}
        try:
            X = sm.add_constant(np.array(y_pred).reshape(-1,1))
            bp_stat, bp_p, _, _ = het_breuschpagan(y_true - y_pred, X)
            results['heteroscedasticity'] = {'bp_stat': float(bp_stat), 'bp_p': float(bp_p), 'is_homoscedastic': bp_p>0.05}
        except Exception as e:
            results['heteroscedasticity'] = {'error': str(e)}

        seg_mets = []
        for seg in hist['segment'].dropna().unique():
            h = hist[hist['segment']==seg]
            if len(h) < 10: continue
            seg_mets.append({
                'segment': seg,
                'r2': r2_score(h['Venda_Vol'], h['yhat_final']),
                'mape': mape(h['Venda_Vol'], h['yhat_final']),
                'wape': wape(h['Venda_Vol'], h['yhat_final']),
                'rmse': float(np.sqrt(mean_squared_error(h['Venda_Vol'], h['yhat_final']))),
                'mae': mean_absolute_error(h['Venda_Vol'], h['yhat_final']),
                'n_obs': int(len(h))
            })
        results['segment_metrics'] = seg_mets

        try:
            cv_scores = self._time_series_cv(hist)
            results['cross_validation'] = cv_scores
        except Exception as e:
            results['cross_validation'] = {'error': str(e)}
        return results

    @staticmethod
    def _time_series_cv(df: pd.DataFrame, n_splits: int = 3) -> Dict[str, float]:
        """Validação cruzada temporal simples com baseline de médias."""
        d = df.sort_values('Data')
        tscv = TimeSeriesSplit(n_splits=n_splits)
        out = {'r2':[],'mape':[],'rmse':[]}
        vals = d[['UF','CD_HIER_N11','Data','Venda_Vol']]
        for tr_idx, va_idx in tscv.split(vals):
            tr = vals.iloc[tr_idx]; va = vals.iloc[va_idx]
            preds = []
            for (uf, prod), g in va.groupby(['UF','CD_HIER_N11']):
                gt = tr[(tr['UF']==uf)&(tr['CD_HIER_N11']==prod)]
                if not gt.empty:
                    p = gt.tail(6)['Venda_Vol'].mean()
                else:
                    p = tr['Venda_Vol'].mean()
                preds.extend([p]*len(g))
            y_true = va['Venda_Vol'].values
            y_pred = np.array(preds)
            out['r2'].append(r2_score(y_true, y_pred))
            out['mape'].append(mape(y_true, y_pred))
            out['rmse'].append(float(np.sqrt(mean_squared_error(y_true, y_pred))))
        return {k+'_mean': float(np.mean(v)) for k,v in out.items()} | {k+'_std': float(np.std(v)) for k,v in out.items()}

    # ------------------------------------------------------------------
    # 4.6 FORECAST FUTURO HIERÁRQUICO E EXPORTAÇÃO
    # ------------------------------------------------------------------
    def forecast_future_hier(self, base_hist: pd.DataFrame) -> pd.DataFrame:
        """Projeta vendas futuras hierarquicamente (UF → Linha → Grupo → Produto)."""
        if not {'DS_LINHA','DS_GRUPO'}.issubset(set(base_hist.columns)):
            logger.warning("Sem DS_LINHA/DS_GRUPO — forecast futuro hierárquico será pulado.")
            return pd.DataFrame()

        hist_total = (
            base_hist.groupby(['UF','DS_LINHA','Data'], observed=True)['Venda_Vol']
                     .sum().reset_index()
        )
        rows = []
        future_dates = pd.date_range(start=base_hist['Data'].max() + relativedelta(months=1), periods=HORIZONTE, freq='MS')
        for d in future_dates:
            for uf in hist_total['UF'].unique():
                df_uf = hist_total[hist_total['UF']==uf]
                for ln in df_uf['DS_LINHA'].unique():
                    ts_line = df_uf[df_uf['DS_LINHA']==ln].set_index('Data')['Venda_Vol']
                    if len(ts_line) < MIN_HISTORICO_MESES:
                        continue
                    recent = ts_line.tail(WINDOW_BOUND)
                    trend = recent.mean() if not recent.empty else ts_line.mean()
                    seasonal = ts_line[ts_line.index.month == d.month]
                    seasonal_mean = seasonal.mean() if not seasonal.empty else trend
                    val_line = seasonal_mean * (1 + PESO_YOY)

                    lim_line_hist = base_hist[(base_hist['UF']==uf)&(base_hist['DS_LINHA']==ln)&(base_hist['Data']<d)&(base_hist['Data']>=d - relativedelta(months=WINDOW_BOUND))]['Venda_Vol'].sum()
                    lim_line = max(1.0, lim_line_hist) * FATOR_LIMITE

                    hist_line = base_hist[(base_hist['UF']==uf)&(base_hist['DS_LINHA']==ln)&(base_hist['Data']<d)&(base_hist['Data']>=d - relativedelta(months=WINDOW_BOUND))]
                    grupos = hist_line['DS_GRUPO'].unique()
                    bruto = {}
                    for g in grupos:
                        yoy_ini = (d - relativedelta(months=12)).replace(day=1)
                        yoy_fim = yoy_ini + relativedelta(months=1) - pd.Timedelta(days=1)
                        s_yoy = base_hist[(base_hist['UF']==uf)&(base_hist['DS_LINHA']==ln)&(base_hist['DS_GRUPO']==g)&(base_hist['Data']>=yoy_ini)&(base_hist['Data']<=yoy_fim)]['Venda_Vol'].sum()
                        if s_yoy>0:
                            val = s_yoy * (1+PESO_YOY)
                        else:
                            s_roll = hist_line[hist_line['DS_GRUPO']==g]['Venda_Vol'].mean()
                            val = (s_roll if np.isfinite(s_roll) else 0.0) * (1+PESO_YOY)
                        lim_g_hist = hist_line[hist_line['DS_GRUPO']==g]['Venda_Vol'].sum()
                        bruto[g] = min(val, max(1.0, lim_g_hist) * FATOR_LIMITE)

                    total_g = sum(bruto.values()) or 1.0
                    ajuste = min(1.0, lim_line / total_g)
                    for g, v in bruto.items():
                        vg = v * ajuste
                        dfh = hist_line[hist_line['DS_GRUPO']==g]
                        dist = dfh.groupby('CD_HIER_N11', observed=True)['Venda_Vol'].sum().reset_index()
                        tot = dist['Venda_Vol'].sum() or 1.0
                        for _, r in dist.iterrows():
                            rows.append({
                                'UF': uf, 'DS_LINHA': ln, 'DS_GRUPO': g, 'CD_HIER_N11': r['CD_HIER_N11'],
                                'Data': d, 'Ano': d.year, 'MesOrig': d.month,
                                'Venda_Vol_Prevista': max(0.0, vg * (r['Venda_Vol']/tot)), 'Fase': 'Futuro'
                            })
        return pd.DataFrame(rows)

    def export_results(self, hist: pd.DataFrame, validation: Dict[str, Any], future: pd.DataFrame) -> None:
        """Exporta resultados para Excel e gera gráficos de diagnóstico."""
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        CHARTS_LINES_DIR.mkdir(parents=True, exist_ok=True)
        try:
            eng = 'openpyxl'
            with pd.ExcelWriter(OUTPUT_PATH, engine=eng) as writer:
                hist_out = hist.copy()
                hist_out.rename(columns={'Venda_Vol':'Venda_Vol_Real', 'yhat_final':'Venda_Vol_Prevista'}, inplace=True)
                hist_out[['UF','CD_HIER_N11','Ano','Data','Venda_Vol_Real','Venda_Vol_Prevista','segment']].to_excel(writer, sheet_name='Historico', index=False)
                pd.DataFrame([validation.get('global_metrics', {})]).to_excel(writer, sheet_name='Metricas_Globais', index=False)
                pd.DataFrame(validation.get('segment_metrics', [])).to_excel(writer, sheet_name='Metricas_Segmentos', index=False)
                pd.DataFrame(validation.get('linha_metrics', [])).to_excel(writer, sheet_name='Metricas_Linha', index=False)

                md = [{'Segment': seg, **mets} for seg, mets in self.segment_metrics.items()]
                pd.DataFrame(md).to_excel(writer, sheet_name='Detalhes_Modelos', index=False)
                if not future.empty:
                    n_chunks = int(np.ceil(len(future)/EXCEL_MAX_ROWS)) or 1
                    for i, chunk in enumerate(np.array_split(future, n_chunks), 1):
                        chunk.to_excel(writer, sheet_name=f'Futuro_{i}', index=False)
                cap_cols = [c for c in ['cap_hist','cap_upper'] if c in hist.columns]
                if cap_cols:
                    (hist[['UF','CD_HIER_N11'] + cap_cols]
                        .drop_duplicates()
                        .to_excel(writer, sheet_name='Caps_Series', index=False))

        except Exception:
            with pd.ExcelWriter(OUTPUT_PATH) as writer:
                hist_out = hist.copy()
                hist_out.rename(columns={'Venda_Vol':'Venda_Vol_Real', 'yhat_final':'Venda_Vol_Prevista'}, inplace=True)
                hist_out[['UF','CD_HIER_N11','Ano','Data','Venda_Vol_Real','Venda_Vol_Prevista','segment']].to_excel(writer, sheet_name='Historico', index=False)
                pd.DataFrame([validation.get('global_metrics', {})]).to_excel(writer, sheet_name='Metricas_Globais', index=False)
                pd.DataFrame(validation.get('segment_metrics', [])).to_excel(writer, sheet_name='Metricas_Segmentos', index=False)
                md = [{'Segment': seg, **mets} for seg, mets in self.segment_metrics.items()]
                pd.DataFrame(md).to_excel(writer, sheet_name='Detalhes_Modelos', index=False)
                if not future.empty:
                    n_chunks = int(np.ceil(len(future)/EXCEL_MAX_ROWS)) or 1
                    for i, chunk in enumerate(np.array_split(future, n_chunks), 1):
                        chunk.to_excel(writer, sheet_name=f'Futuro_{i}', index=False)

        self._plots_diagnostics(hist)
        if {'DS_LINHA'} <= set(hist.columns):
            self._plots_lines(hist, future)

    def _plots_diagnostics(self, hist: pd.DataFrame) -> None:
        """Gera gráficos de diagnóstico de resíduos."""
        y = pd.to_numeric(hist['Venda_Vol'], errors='coerce')
        yhat = pd.to_numeric(hist['yhat_final'], errors='coerce')
        res = (y - yhat).replace([np.inf, -np.inf], np.nan).dropna()

        plt.figure(figsize=(12,8))
        plt.subplot(2,2,1)
        plt.scatter(yhat, y - yhat, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.title('Resíduos vs Preditos'); plt.xlabel('Preditos'); plt.ylabel('Resíduos')

        plt.subplot(2,2,2)
        if len(res) > 3:
            stats.probplot(res, dist='norm', plot=plt)
        plt.title('QQ-plot Resíduos')

        plt.subplot(2,2,3)
        if len(res) > 1:
            plt.hist(res, bins=50, alpha=0.7, density=True)
        plt.title('Distribuição Resíduos')

        plt.subplot(2,2,4)
        mask = y.notna() & yhat.notna()
        plt.scatter(y[mask], yhat[mask], alpha=0.5)
        if mask.any():
            mn = float(np.nanmin([y[mask].min(), yhat[mask].min()]))
            mx = float(np.nanmax([y[mask].max(), yhat[mask].max()]))
            if np.isfinite(mn) and np.isfinite(mx):
                plt.plot([mn, mx], [mn, mx], 'r--')
        plt.title('Real vs Predito'); plt.xlabel('Real'); plt.ylabel('Predito')

        plt.tight_layout()
        plt.savefig(CHARTS_DIR / 'diagnosticos_residuos.png', dpi=300, bbox_inches='tight')
        plt.close('all')

    def _plots_lines(self, hist: pd.DataFrame, future: pd.DataFrame) -> None:
        """Produz gráficos de linha por DS_LINHA com histórico e previsão."""
        hist_valid = hist[hist['yhat_final'].notna()].copy()

        h = (hist_valid.groupby(['DS_LINHA','Data'], observed=True)
                      .agg(Venda_Real=('Venda_Vol','sum'),
                           Venda_PrevHist=('yhat_final','sum'))
                      .reset_index())

        if not future.empty and 'Venda_Vol_Prevista' in future.columns:
            f = (future.groupby(['DS_LINHA','Data'], observed=True)['Venda_Vol_Prevista']
                        .sum()
                        .reset_index())
        else:
            f = pd.DataFrame(columns=['DS_LINHA','Data','Venda_Vol_Prevista'])

        top10 = h.groupby('DS_LINHA')['Venda_Real'].sum().nlargest(10).index
        for ln in top10:
            hh = h[h['DS_LINHA'] == ln].sort_values('Data')
            ff = f[f['DS_LINHA'] == ln].sort_values('Data')
            plt.figure(figsize=(10, 4))
            plt.plot(hh['Data'], hh['Venda_Real'], 'o-')
            plt.plot(hh['Data'], hh['Venda_PrevHist'], 'x--')
            if not ff.empty:
                plt.plot(ff['Data'], ff['Venda_Vol_Prevista'], 's-.')
            plt.title(f'Vendas Mensais - {ln}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            safe = _sanitize_filename(ln)
            plt.savefig(CHARTS_LINES_DIR / f'vendas_{safe}.png')
            plt.close()

    # ------------------------------------------------------------------
    # 4.7 EXECUÇÃO DO PIPELINE COMPLETO
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Executa todo o pipeline de previsão."""
        logger.info("=== INÍCIO PIPELINE  ===")
        df = self.load_and_preprocess()

        logger.info("Classificando ABC...")
        abc_df = self.classify_abc_advanced(df)
        logger.info("Classificando séries...")
        ts_df  = self.classify_time_series_advanced(df)

        logger.info("Criando segmentos...")
        df = self.create_segments_hierarchical(df, abc_df, ts_df)

        logger.info("Aplicando transformações por segmento...")
        df = self.apply_target_transformations(df)

        logger.info("Split treino/teste...")
        train, test = split_train_test(df)
        logger.info(f"Treino: {len(train):,} | Teste: {len(test):,}")

        logger.info("Modelagem por segmento...")
        tr, te, seg_mets = self.fit_models_with_fallback(train, test)
        self.segment_metrics = seg_mets

        def _ema6_baseline(series: pd.Series) -> float:
            s = series.dropna().astype(float)
            if s.empty:
                return np.nan
            w = np.exp(np.linspace(0, 1, min(6, len(s))))
            return float(np.average(s.tail(len(w)), weights=w))

        for (uf, prod), g in tr.groupby(['UF', 'CD_HIER_N11']):
            if not g['yhat'].notna().any():
                fb = _ema6_baseline(g.sort_values('Data')['Venda_Vol'])
                if np.isfinite(fb):
                    tr.loc[g.index, 'yhat'] = fb

        logger.info("Recalibração WLS (variance smoothing) sem leakage...")

        wls_tr, diag = self.apply_wls_smoothing(
            tr[['UF','CD_HIER_N11','Data','Venda_Vol','yhat','segment']].copy()
        )
        use_wls = diag.get('bp_wls_smooth_p', 0) >= diag.get('bp_orig_p', 0)

        tr = tr.merge(
            wls_tr[['UF','CD_HIER_N11','Data','yhat_wls_smooth']],
            on=['UF','CD_HIER_N11','Data'], how='left'
        )
        tr['yhat_final'] = np.where(
            use_wls & tr['yhat_wls_smooth'].notna(),
            tr['yhat_wls_smooth'], tr['yhat']
        ).astype(float)

        Xtr = sm.add_constant(tr['yhat_final'].fillna(0.0).astype(float))
        ols_tr = sm.OLS(tr['Venda_Vol'].astype(float), Xtr).fit()

        params = ols_tr.params
        if isinstance(params, pd.Series):
            beta_const = float(params['const']) if 'const' in params.index else 0.0
            if 'yhat_final' in params.index:
                beta_slope = float(params['yhat_final'])
            elif 'x1' in params.index:
                beta_slope = float(params['x1'])
            else:
                beta_slope = float(params.iloc[-1])
        else:
            arr = np.asarray(params).ravel()
            beta_const = float(arr[0]) if arr.size >= 1 else 0.0
            beta_slope = float(arr[1]) if arr.size >= 2 else 1.0

        te_in = te['yhat'].fillna(0.0).astype(float)
        te['yhat_final'] = beta_const + beta_slope * te_in

        def _series_fallback(s: pd.Series) -> float:
            s = s.dropna().astype(float)
            if s.empty:
                return np.nan
            w = np.exp(np.linspace(0, 1, min(6, len(s))))
            return float(np.average(s.tail(len(w)), weights=w))

        te = te.sort_values('Data')
        for (uf, prod), g in te.groupby(['UF','CD_HIER_N11']):
            mask = (te['UF']==uf) & (te['CD_HIER_N11']==prod)
            yhat_curr = te.loc[mask, 'yhat_final'].values.astype(float)
            if (np.all(~np.isfinite(yhat_curr))) or (np.all(yhat_curr == 0.0)) or (np.isnan(yhat_curr).any()):
                hist_train_series = tr[(tr['UF']==uf)&(tr['CD_HIER_N11']==prod)].sort_values('Data')
                fb = _series_fallback(hist_train_series['Venda_Vol'])
                if np.isfinite(fb):
                    te.loc[mask, 'yhat_final'] = fb

        hist = pd.concat([tr, te], ignore_index=True)

        caps_hist = self.compute_caps_per_series(train).rename(columns={'cap_upper':'cap_hist'})
        hist = hist.merge(caps_hist, on=['UF','CD_HIER_N11'], how='left')
        mask_cap = hist['cap_hist'].notna()
        hist.loc[mask_cap, 'yhat_final'] = np.minimum(
            hist.loc[mask_cap, 'yhat_final'].astype(float),
            hist.loc[mask_cap, 'cap_hist'].astype(float)
        ).astype(float)
        hist['yhat_final'] = hist['yhat_final'].clip(lower=0.0).astype(float)

        logger.info("Validando modelo...")
        validation = self.validate(hist)

        linha_metrics = validation.get('linha_metrics', [])
        if linha_metrics:
            mape_linhas = np.nanmean([m['mape'] for m in linha_metrics if 'mape' in m])
            logger.info(f"MAPE médio por Linha: {mape_linhas:.2f}%")

        logger.info("Gerando forecast futuro hierárquico...")
        future = self.forecast_future_hier(df)
        if not future.empty:
            logger.info("Aplicando winsorization por série no FUTURO (cap = cap_upper * CAP_FUT_MULT)...")
            fut_caps = self.compute_caps_per_series(train)
            if not fut_caps.empty:
                fut_caps = fut_caps.rename(columns={'cap_upper': 'cap_upper_future'})
                fut_caps['cap_upper_future'] = fut_caps['cap_upper_future'].astype(float) * float(CAP_FUT_MULT)

                future = future.merge(
                    fut_caps[['UF','CD_HIER_N11','cap_upper_future']],
                    on=['UF','CD_HIER_N11'], how='left'
                )
                cap_fut_vec = future['cap_upper_future'].where(future['cap_upper_future'].notna(), np.inf).astype(float)
                future['Venda_Vol_Prevista'] = np.minimum(future['Venda_Vol_Prevista'].astype(float), cap_fut_vec)
                future['Venda_Vol_Prevista'] = future['Venda_Vol_Prevista'].clip(lower=0).astype(float)
            else:
                logger.warning("Sem caps para FUTURO — winsorization futura pulada.")

        logger.info(f"Exportando resultados para {OUTPUT_PATH} ...")
        self.export_results(hist, validation, future)
        logger.info(f"Gráficos de diagnóstico: {CHARTS_DIR / 'diagnosticos_residuos.png'}")
        logger.info(f"Gráficos salvos em {CHARTS_DIR} e {CHARTS_LINES_DIR}")
        logger.info("=== PIPELINE FINALIZADO ===")


# =============================================================================
# 5. PONTO DE ENTRADA
# =============================================================================
def main():
    """Função de entrada para execução direta do script."""
    try:
        f = Forecaster(FILE_PATH)
        f.run()
        logger.info("🎉 Processo concluído com sucesso!")
    except Exception as e:
        logger.exception(f"Erro no pipeline: {e}")
        raise


if __name__ == '__main__':
    main()
