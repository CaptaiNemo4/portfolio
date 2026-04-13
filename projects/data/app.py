"""
Data Science Portfolio 

Two projects showcased:
1. Bayesian Volatility Modeling (GARCH vs Stochastic Volatility)
2. Portfolio Optimization (Efficient Frontier, Max Sharpe, GMV, Risk Parity)

"""

from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Type
from functools import wraps


# custom exceptions for better error handling and debugging

class DataLoadError(Exception):
    """Raised when data files cannot be loaded."""
    def __init__(self, filename: str, reason: str = ""):
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to load '{filename}': {reason}")


class ValidationError(Exception):
    """Raised when data validation fails."""
    def __init__(self, missing_columns: List[str]):
        self.missing_columns = missing_columns
        super().__init__(f"Missing required columns: {missing_columns}")


class StrategyNotFoundError(Exception):
    """Raised when a requested strategy doesn't exist."""
    pass


# class decorator for singleton pattern to ensure only one instance of AppController exists

def singleton(cls):
    """Class decorator ensuring only one instance exists."""
    instances = {}
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance


# data loaders with Template Method pattern

class DataLoader(ABC):
    """Abstract base class for data loading strategies."""

    @abstractmethod
    def load(self, filename: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        pass

    def safe_load(self, filename: str, required_columns: List[str] = None) -> pd.DataFrame:
        """Template Method — fixed algorithm, subclasses override steps."""
        try:
            df = self.load(filename)
            if required_columns:
                self.validate(df, required_columns)
            return df
        except FileNotFoundError:
            raise DataLoadError(filename, "File not found")
        except ValidationError as e:
            raise DataLoadError(filename, str(e))


class CSVDataLoader(DataLoader):
    """Concrete CSV loader. Inherits from DataLoader."""

    def __init__(self, base_path: str):
        self._base_path = base_path

    @property
    def base_path(self) -> str:
        return self._base_path

    def load(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(self._base_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Not found: {filepath}")
        df = pd.read_csv(filepath)
        for col in ['date', 'Date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    def validate(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValidationError(missing)
        return True

    def __repr__(self) -> str:
        return f"CSVDataLoader('{self._base_path}')"


# data managers for each project, using lazy loading and caching to optimize performance

class ProjectDataManager(ABC):
    """Abstract base for project data managers."""

    def __init__(self, loader: DataLoader):
        self._loader = loader
        self._loaded = False

    @abstractmethod
    def load_all(self) -> None:
        pass

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_all()
            self._loaded = True


class VolatilityDataManager(ProjectDataManager):
    """Manages Volatility Modeling data."""

    REQUIRED_COLUMNS = ['date', 'actual_volatility', 'garch_predicted',
                        'sv_predicted', 'weekly_return']

    def __init__(self, loader: DataLoader):
        super().__init__(loader)
        self._train_data: Optional[pd.DataFrame] = None
        self._test_data: Optional[pd.DataFrame] = None
        self._parameters: Optional[pd.DataFrame] = None
        self._mse: Optional[pd.DataFrame] = None

    def load_all(self) -> None:
        self._train_data = self._loader.safe_load("train_data.csv", self.REQUIRED_COLUMNS)
        self._test_data = self._loader.safe_load("test_data.csv", self.REQUIRED_COLUMNS)
        self._parameters = self._loader.safe_load("model_parameters.csv")
        self._mse = self._loader.safe_load("mse_comparison.csv")
        self._loaded = True

    @property
    def train_data(self):
        self._ensure_loaded()
        return self._train_data

    @property
    def test_data(self):
        self._ensure_loaded()
        return self._test_data

    @property
    def parameters(self):
        self._ensure_loaded()
        return self._parameters

    @property
    def mse(self):
        self._ensure_loaded()
        return self._mse

    def get_dataset(self, name: str) -> pd.DataFrame:
        if name == "Train (2013\u20132017)":
            return self.train_data
        return self.test_data

    def __del__(self):
        self._train_data = None
        self._test_data = None


class PortfolioDataManager(ProjectDataManager):
    """Manages Portfolio Optimization data."""

    def __init__(self, loader: DataLoader):
        super().__init__(loader)
        self._cumulative_returns: Optional[pd.DataFrame] = None
        self._weights: Optional[pd.DataFrame] = None
        self._metrics: Optional[pd.DataFrame] = None
        self._efficient_frontier: Optional[pd.DataFrame] = None
        self._asset_metrics: Optional[pd.DataFrame] = None

    def load_all(self) -> None:
        self._cumulative_returns = self._loader.safe_load("cumulative_returns.csv")
        self._weights = self._loader.safe_load("strategy_weights.csv")
        self._metrics = self._loader.safe_load("performance_metrics.csv")
        self._efficient_frontier = self._loader.safe_load("efficient_frontier.csv")
        self._asset_metrics = self._loader.safe_load("asset_metrics.csv")
        self._loaded = True

    @property
    def cumulative_returns(self):
        self._ensure_loaded()
        return self._cumulative_returns

    @property
    def weights(self):
        self._ensure_loaded()
        return self._weights

    @property
    def metrics(self):
        self._ensure_loaded()
        return self._metrics

    @property
    def efficient_frontier(self):
        self._ensure_loaded()
        return self._efficient_frontier

    @property
    def asset_metrics(self):
        self._ensure_loaded()
        return self._asset_metrics

    def __del__(self):
        self._cumulative_returns = None
        self._weights = None


# volatility models using Strategy Pattern to encapsulate different modeling approaches

class VolatilityModel(ABC):
    def __init__(self, name: str, color: str):
        self._name = name
        self._color = color
        self._parameters: Dict[str, float] = {}

    @property
    def name(self): return self._name
    @property
    def color(self): return self._color
    @property
    def parameters(self): return self._parameters.copy()

    @abstractmethod
    def get_prediction_column(self) -> str: pass
    @abstractmethod
    def get_description(self) -> str: pass

    def load_parameters(self, params_df):
        for _, row in params_df[params_df['model'] == self._name].iterrows():
            self._parameters[row['parameter']] = row['estimate']

    def __str__(self): return f"{self._name} Model"
    def __repr__(self): return f"{self.__class__.__name__}('{self._name}')"


class GARCHModel(VolatilityModel):
    def __init__(self):
        super().__init__("GARCH(1,1)", "#2563eb")
    def get_prediction_column(self): return "garch_predicted"
    def get_description(self):
        return "\u03c3\u00b2\u209c = \u03b1\u2080 + \u03b1\u2081\u00b7a\u00b2\u209c\u208b\u2081 + \u03b2\u2081\u00b7\u03c3\u00b2\u209c\u208b\u2081 \u2014 models variance as a deterministic function of past shocks and past variance."


class SVModel(VolatilityModel):
    def __init__(self):
        super().__init__("SV", "#dc2626")
    def get_prediction_column(self): return "sv_predicted"
    def get_description(self):
        return "y\u209c = \u03b5\u209c\u00b7exp(h\u209c/2), h\u209c\u208a\u2081 = \u03bc + \u03c6(h\u209c\u2212\u03bc) + \u03b4\u209c\u03c3 \u2014 treats log-volatility as a latent AR(1) process."


# optimization strategies using Strategy Pattern to encapsulate different portfolio construction approaches

class OptimizationStrategy(ABC):
    def __init__(self, name: str, display_name: str, color: str):
        self._name = name
        self._display_name = display_name
        self._color = color

    @property
    def name(self): return self._name
    @property
    def display_name(self): return self._display_name
    @property
    def color(self): return self._color

    @abstractmethod
    def get_description(self) -> str: pass
    def __str__(self): return self._display_name


class EfficientFrontierStrategy(OptimizationStrategy):
    _COLORS = {1: "#3b82f6", 2: "#f97316", 3: "#22c55e"}
    _DESCS = {1: "Conservative \u2014 low risk, low return",
              2: "Moderate \u2014 balanced risk/return",
              3: "Aggressive \u2014 high risk, high return"}
    def __init__(self, point: int):
        super().__init__(f"EF_{point}", f"EF point {point}", self._COLORS.get(point, "#6b7280"))
        self._point = point
    def get_description(self): return self._DESCS.get(self._point, "")

class MaxSharpeStrategy(OptimizationStrategy):
    def __init__(self): super().__init__("Max_Sharpe", "Max Sharpe ratio", "#a855f7")
    def get_description(self): return "Maximizes risk-adjusted return."

class GMVStrategy(OptimizationStrategy):
    def __init__(self): super().__init__("GMV", "Global min variance", "#6366f1")
    def get_description(self): return "Minimizes total portfolio volatility."

class RiskParityStrategy(OptimizationStrategy):
    def __init__(self): super().__init__("Risk_Parity", "Risk parity", "#ec4899")
    def get_description(self): return "Equalizes risk contribution of each asset."

class EqualWeightsStrategy(OptimizationStrategy):
    def __init__(self): super().__init__("Equal_Weights", "Equal weights", "#14b8a6")
    def get_description(self): return "Naive benchmark \u2014 equal weight to all assets."

class SPYBenchmark(OptimizationStrategy):
    def __init__(self): super().__init__("SPY", "S&P 500", "#64748b")
    def get_description(self): return "S&P 500 index \u2014 market benchmark."


# strategy factory to create strategy instances based on user selection, demonstrating Factory Pattern

class StrategyFactory:
    """Creates strategy objects. Demonstrates Factory Pattern + @classmethod."""
    _registry: Dict[str, Type[OptimizationStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[OptimizationStrategy]):
        cls._registry[name] = strategy_class

    @classmethod
    def create(cls, name: str, **kwargs) -> OptimizationStrategy:
        if name not in cls._registry:
            raise StrategyNotFoundError(f"Unknown: {name}")
        return cls._registry[name](**kwargs)

    @classmethod
    def create_all_default(cls) -> Dict[str, OptimizationStrategy]:
        return {
            "ef1": EfficientFrontierStrategy(1), "ef2": EfficientFrontierStrategy(2),
            "ef3": EfficientFrontierStrategy(3), "max_sharpe": MaxSharpeStrategy(),
            "gmv": GMVStrategy(), "risk_parity": RiskParityStrategy(),
            "equal": EqualWeightsStrategy(), "spy": SPYBenchmark()
        }

StrategyFactory.register("max_sharpe", MaxSharpeStrategy)
StrategyFactory.register("gmv", GMVStrategy)
StrategyFactory.register("risk_parity", RiskParityStrategy)
StrategyFactory.register("equal", EqualWeightsStrategy)


# visualization classes for generating Plotly charts, using a base class to share common layout settings and styling

class BaseVisualizer(ABC):
    """Base with shared layout template."""
    @staticmethod
    def _apply_layout(fig, title, xaxis, yaxis, height=280, **kw):
        fig.update_layout(title=dict(text=title, font=dict(size=11), y=0.98),
            xaxis_title=dict(text=xaxis, font=dict(size=10)),
            yaxis_title=dict(text=yaxis, font=dict(size=10)),
            template='plotly_white', hovermode='x unified',
            legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5, font=dict(size=9)),
            margin=dict(l=45, r=5, t=30, b=50), height=height, **kw)


class VolatilityVisualizer(BaseVisualizer):
    @staticmethod
    def create_volatility_chart(df, models, title):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['actual_volatility'], mode='lines',
            name='Actual', line=dict(color='#1e293b', width=1.5)))
        for m in models:
            col = m.get_prediction_column()
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df['date'], y=df[col], mode='lines',
                    name=f'{m.name}', line=dict(color=m.color, width=2, dash='dash')))
        BaseVisualizer._apply_layout(fig, title, 'Date', 'Volatility')
        return pio.to_html(fig, include_plotlyjs=True, full_html=False)

    @staticmethod
    def create_mse_chart(mse_df, selected_models):
        colors = {'GARCH(1,1)': '#2563eb', 'SV': '#dc2626'}
        fig = go.Figure()
        for m in selected_models:
            d = mse_df[mse_df['model'] == m]
            fig.add_trace(go.Bar(x=d['dataset'], y=d['mse'], name=m,
                marker_color=colors.get(m, '#6b7280')))
        BaseVisualizer._apply_layout(fig, 'MSE comparison', 'Dataset', 'MSE', 400, barmode='group')
        return pio.to_html(fig, include_plotlyjs=True, full_html=False)

    @staticmethod
    def create_returns_chart(df):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['date'], y=df['weekly_return'],
            marker_color=np.where(df['weekly_return'] >= 0, '#22c55e', '#ef4444').tolist()))
        BaseVisualizer._apply_layout(fig, 'IVV weekly returns', 'Date', 'Return', 350, yaxis_tickformat='.1%')
        return pio.to_html(fig, include_plotlyjs=True, full_html=False)


class PortfolioVisualizer(BaseVisualizer):
    @staticmethod
    def create_cumulative_returns_chart(cum_df, strategies):
        fig = go.Figure()
        for s in strategies:
            if s.name in cum_df.columns:
                fig.add_trace(go.Scatter(x=cum_df['Date'], y=cum_df[s.name], mode='lines',
                    name=s.display_name, line=dict(color=s.color, width=2)))
        BaseVisualizer._apply_layout(fig, 'Cumulative returns (2021-2024)',
            'Date', 'Cumulative Return', yaxis_tickformat='.0%')
        return pio.to_html(fig, include_plotlyjs=True, full_html=False)

    @staticmethod
    def create_efficient_frontier_chart(ef_df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ef_df['volatility'], y=ef_df['return'],
            mode='lines+markers', name='Efficient frontier',
            line=dict(color='#3b82f6', width=2), marker=dict(size=6)))
        for i, idx in enumerate([3, 12, 18]):
            if idx < len(ef_df):
                fig.add_trace(go.Scatter(x=[ef_df.iloc[idx]['volatility']],
                    y=[ef_df.iloc[idx]['return']], mode='markers', name=f'Point {i+1}',
                    marker=dict(size=10, symbol='star', line=dict(width=1, color='#1e293b'))))
        BaseVisualizer._apply_layout(fig, 'Efficient frontier', 'Volatility', 'Return',
            450, xaxis_tickformat='.0%', yaxis_tickformat='.0%')
        return pio.to_html(fig, include_plotlyjs=True, full_html=False)

    @staticmethod
    def create_sharpe_chart(metrics_df):
        colors = ['#3b82f6','#f97316','#22c55e','#a855f7','#6366f1','#14b8a6','#ec4899','#64748b']
        fig = go.Figure()
        fig.add_trace(go.Bar(x=metrics_df['strategy'], y=metrics_df['sharpe_ratio'],
            marker_color=colors[:len(metrics_df)],
            text=metrics_df['sharpe_ratio'].round(2), textposition='auto'))
        BaseVisualizer._apply_layout(fig, 'Sharpe ratios', 'Strategy', 'Sharpe Ratio', 400)
        return pio.to_html(fig, include_plotlyjs=True, full_html=False)

    @staticmethod
    def create_weights_chart(weights_df, strategy_name):
        if strategy_name not in weights_df.columns:
            return "<p>No data.</p>"
        w = weights_df[['asset', strategy_name]].copy()
        w = w[w[strategy_name] > 0.001].sort_values(strategy_name, ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=w[strategy_name], y=w['asset'], orientation='h',
            marker_color='#3b82f6',
            text=(w[strategy_name]*100).round(1).astype(str)+'%', textposition='auto'))
        BaseVisualizer._apply_layout(fig, f'Allocation \u2014 {strategy_name}', 'Weight', '', 400,
            xaxis_tickformat='.0%')
        return pio.to_html(fig, include_plotlyjs=True, full_html=False)

    @staticmethod
    def create_risk_return_scatter(metrics_df):
        colors = ['#3b82f6','#f97316','#22c55e','#a855f7','#6366f1','#14b8a6','#ec4899','#64748b']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_df['annualized_volatility'],
            y=metrics_df['annualized_return'], mode='markers+text',
            text=metrics_df['strategy'], textposition='top right',
            marker=dict(size=9, color=colors[:len(metrics_df)])))
        BaseVisualizer._apply_layout(fig, 'Risk vs return', 'Volatility', 'Return', 450,
            xaxis_tickformat='.0%', yaxis_tickformat='.0%')
        return pio.to_html(fig, include_plotlyjs=True, full_html=False)


# ui builder

class UIBuilder:
    @staticmethod
    def _vol_tab():
        return ui.nav_panel("Volatility Modeling", ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Controls"),
                ui.input_select("vol_dataset", "Dataset:",
                    choices=["Train (2013\u20132017)", "Test (2018\u2013Present)"]),
                ui.input_checkbox_group("vol_models", "Models:",
                    choices={"garch": "GARCH(1,1)", "sv": "Stochastic Volatility"},
                    selected=["garch", "sv"]),
                ui.input_switch("vol_show_returns", "Show weekly returns", value=False),
                ui.hr(), ui.output_ui("vol_model_info"), width=200),
            ui.output_ui("vol_chart"), ui.output_ui("vol_returns_chart"),
            ui.hr(), ui.h4("Model Comparison"), ui.output_ui("vol_mse_chart"),
            ui.hr(), ui.h4("Parameters"), ui.output_ui("vol_params_table")))

    @staticmethod
    def _port_tab():
        return ui.nav_panel("Portfolio Optimization", ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Controls"),
                ui.input_checkbox_group("port_strategies", "Strategies:", choices={
                    "ef1": "EF Pt 1 (Conservative)", "ef2": "EF Pt 2 (Moderate)",
                    "ef3": "EF Pt 3 (Aggressive)", "max_sharpe": "Max Sharpe",
                    "gmv": "Global Min Variance", "risk_parity": "Risk Parity",
                    "equal": "Equal Weights", "spy": "S&P 500"},
                    selected=["ef2", "max_sharpe", "equal", "spy"]),
                ui.hr(),
                ui.input_select("port_weights_strategy", "Show weights for:",
                    choices=["EF_1","EF_2","EF_3","Max_Sharpe","GMV","Risk_Parity","Equal_Weights"]),
                ui.hr(), ui.output_ui("port_strategy_info"), width=200),
            ui.output_ui("port_cumulative_chart"),
            ui.hr(), ui.h4("Efficient Frontier"), ui.output_ui("port_ef_chart"),
            ui.hr(), ui.h4("Evaluation"),
            ui.layout_columns(ui.output_ui("port_sharpe_chart"),
                              ui.output_ui("port_riskreturn_chart"), col_widths=[6,6]),
            ui.hr(), ui.h4("Asset Allocation"), ui.output_ui("port_weights_chart")))

    @staticmethod
    def build():
        return ui.page_fluid(
            ui.tags.style("""
                body { font-size: 12px; padding: 0.3rem; }
                h2 { font-size: 1.1rem; margin-bottom: 0.2rem; }
                h4 { font-size: 0.9rem; margin-bottom: 0.2rem; }
                .sidebar { font-size: 11px; padding: 0.4rem; }
                .form-group { margin-bottom: 0.3rem; }
                hr { margin: 0.3rem 0; }
                p { margin-bottom: 0.2rem; font-size: 11px; }
                .nav-tabs .nav-link { font-size: 12px; padding: 0.3rem 0.8rem; }
                .checkbox label, .form-check-label { font-size: 11px; }
                select.form-select { font-size: 11px; padding: 0.2rem 0.4rem; }
            """),
            ui.tags.style(".hide-tab { display: none !important; }"),
            ui.tags.script("""
                document.addEventListener('DOMContentLoaded', function() {
                    var params = new URLSearchParams(window.location.search);
                    var tab = params.get('tab');
                    if (tab) {
                        setTimeout(function() {
                            var tabs = document.querySelectorAll('.nav-tabs .nav-link');
                            tabs.forEach(function(t, i) {
                                if (tab === 'vol' && i === 1) { t.classList.add('hide-tab'); }
                                if (tab === 'port' && i === 0) { t.classList.add('hide-tab'); }
                                if (tab === 'vol' && i === 0) { t.click(); }
                                if (tab === 'port' && i === 1) { t.click(); }
                            });
                        }, 500);
                    }
                });
            """),
            ui.h2("Chorobek Sheranov — Data Science Portfolio"),
            ui.p("Interactive exploration of quantitative finance projects."),
            ui.navset_tab(UIBuilder._vol_tab(), UIBuilder._port_tab()))


# app controller to manage data loading, model instantiation, and server logic, using singleton pattern to ensure only one instance exists throughout the app lifecycle

@singleton
class AppController:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        loader = CSVDataLoader(data_dir)
        self._vol_data = VolatilityDataManager(loader)
        self._port_data = PortfolioDataManager(loader)
        try:
            self._vol_data.load_all()
            self._port_data.load_all()
        except DataLoadError as e:
            print(f"Warning: {e}")

        self._vol_models: Dict[str, VolatilityModel] = {"garch": GARCHModel(), "sv": SVModel()}
        for m in self._vol_models.values():
            m.load_parameters(self._vol_data.parameters)

        self._strategies = StrategyFactory.create_all_default()
        self._vol_viz = VolatilityVisualizer()
        self._port_viz = PortfolioVisualizer()

    def get_ui(self): return UIBuilder.build()

    def create_server(self, input, output, session):
        # -- Volatility --
        @reactive.Calc
        def vol_data(): return self._vol_data.get_dataset(input.vol_dataset())
        @reactive.Calc
        def vol_models(): return [self._vol_models[k] for k in input.vol_models() if k in self._vol_models]

        @output
        @render.ui
        def vol_chart():
            return ui.HTML(self._vol_viz.create_volatility_chart(
                vol_data(), vol_models(), f"Actual vs predicted \u2014 {input.vol_dataset()}"))
        @output
        @render.ui
        def vol_returns_chart():
            if not input.vol_show_returns(): return ui.div()
            return ui.HTML(self._vol_viz.create_returns_chart(vol_data()))
        @output
        @render.ui
        def vol_mse_chart():
            names = [self._vol_models[k].name for k in input.vol_models() if k in self._vol_models]
            if not names: return ui.p("Select a model.")
            return ui.HTML(self._vol_viz.create_mse_chart(self._vol_data.mse, names))
        @output
        @render.ui
        def vol_params_table():
            names = [self._vol_models[k].name for k in input.vol_models() if k in self._vol_models]
            if not names: return ui.p("Select a model.")
            p = self._vol_data.parameters
            return ui.HTML(p[p['model'].isin(names)].to_html(index=False, classes='table table-striped'))
        @output
        @render.ui
        def vol_model_info():
            ms = vol_models()
            if not ms: return ui.p("Select a model.")
            return ui.div(*[ui.div(ui.strong(str(m)), ui.p(m.get_description()),
                style="margin-bottom:10px;") for m in ms])

        # -- Portfolio --
        @reactive.Calc
        def port_strats(): return [self._strategies[k] for k in input.port_strategies() if k in self._strategies]

        @output
        @render.ui
        def port_cumulative_chart():
            s = port_strats()
            if not s: return ui.p("Select a strategy.")
            return ui.HTML(self._port_viz.create_cumulative_returns_chart(self._port_data.cumulative_returns, s))
        @output
        @render.ui
        def port_ef_chart():
            return ui.HTML(self._port_viz.create_efficient_frontier_chart(self._port_data.efficient_frontier))
        @output
        @render.ui
        def port_sharpe_chart():
            return ui.HTML(self._port_viz.create_sharpe_chart(self._port_data.metrics))
        @output
        @render.ui
        def port_riskreturn_chart():
            return ui.HTML(self._port_viz.create_risk_return_scatter(self._port_data.metrics))
        @output
        @render.ui
        def port_weights_chart():
            return ui.HTML(self._port_viz.create_weights_chart(self._port_data.weights, input.port_weights_strategy()))
        @output
        @render.ui
        def port_strategy_info():
            ss = port_strats()
            if not ss: return ui.p("Select a strategy.")
            return ui.div(*[ui.div(ui.strong(str(s)), ui.p(s.get_description()),
                style="margin-bottom:10px;") for s in ss])


# run the app

controller = AppController()
app_ui = controller.get_ui()
server = controller.create_server
app = App(app_ui, server)
