# economic_analyzer.py
import pandas as pd
import numpy as np


class EconomicAnalyzer:
    def __init__(self, df_orders, df_economic):
        self.df_orders = df_orders
        self.df_economic = df_economic

    def analyze(self):
        # 1️ Preparar datasets
        df_orders = self.df_orders.copy()
        df_econ = self.df_economic.copy()

        # Convertir fechas a periodo mensual en ambos datasets
        df_orders["order_purchase_timestamp"] = pd.to_datetime(df_orders["order_purchase_timestamp"], errors="coerce")
        df_orders["year_month"] = df_orders["order_purchase_timestamp"].dt.to_period("M").astype(str)

        # Asegurar formato de fecha en df_econ
        if "date" in df_econ.columns:
            df_econ["date"] = pd.to_datetime(df_econ["date"], errors="coerce")
            df_econ["year_month"] = df_econ["date"].dt.to_period("M").astype(str)

        # 2️ Volumen mensual de pedidos
        monthly_orders = (
            df_orders.groupby("year_month")
            .size()
            .reset_index(name="orders_count")
            .sort_values("year_month")
        )

        # 3️ Merge temporal con indicadores económicos
        joined = monthly_orders.merge(df_econ, on="year_month", how="left")

        # 4️ Limpieza de datos nulos
        # Opción 1: eliminar meses sin indicadores (más estricta)
        joined = joined.dropna(subset=["econ_act", "inflation", "interest_rate"], how="all")

        # Opción 2: interpolar valores faltantes (más continua)
        joined = joined.sort_values("year_month")
        joined = joined.ffill().bfill()

        # 5️ Calcular correlaciones
        corr = joined.corr(numeric_only=True)["orders_count"].drop("orders_count", errors="ignore").to_dict()

        # 6️ Tendencia del volumen mensual
        joined["month_index"] = np.arange(len(joined))
        if len(joined) > 1:
            slope, intercept = np.polyfit(joined["month_index"], joined["orders_count"], 1)
            trend = "increasing" if slope > 0 else "decreasing"
        else:
            slope, trend = 0, "stable"

        # 7️ Devolver resultados
        return {
            "monthly_volumes": monthly_orders.to_dict(orient="records"),
            "joined_monthly": joined.to_dict(orient="records"),
            "trend_estimates": {"slope": slope, "trend": trend},
            "econ_correlations_with_orders": corr,
        }
