# etl/processing/metric_calculator.py
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

class MetricCalculator:
    """
    Calcula métricas generales, delivery stats, correlaciones y series temporales.
    """

    def __init__(self, df_orders, df_items, df_customers, df_geolocation, df_economic, df_products=None):
        self.df_orders = df_orders.copy() if df_orders is not None else pd.DataFrame()
        self.df_items = df_items.copy() if df_items is not None else pd.DataFrame()
        self.df_customers = df_customers.copy() if df_customers is not None else pd.DataFrame()
        self.df_geolocation = df_geolocation.copy() if df_geolocation is not None else pd.DataFrame()
        self.df_economic = df_economic.copy() if df_economic is not None else pd.DataFrame()
        self.df_products = df_products.copy() if df_products is not None else pd.DataFrame()

    def calculate_all(self):
        results = {}
        # global metrics
        results["metrics"] = self._generate_metrics()

        # delivery analysis: compute delivery days, classify speed, stats global
        results["delivery_stats"] = self._analyze_delivery_performance()

        # economic analysis
        results["economic_analysis"] = self._analyze_economic_relations_and_trend()

        results["timestamp"] = datetime.utcnow().isoformat()
        return results

    def _generate_metrics(self):
        total_customers = len(self.df_customers) if not self.df_customers.empty else 0
        total_items = len(self.df_items) if not self.df_items.empty else 0
        items_per_customer = round(total_items/total_customers,2) if total_customers>0 else None

        return {
            "total_customers": int(total_customers),
            "total_items": int(total_items),
            "items_per_customer_avg": items_per_customer
        }

    def _analyze_delivery_performance(self):
        df = self.df_orders.copy()
        required = ["order_purchase_timestamp","order_delivered_customer_date"]
        for c in required:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        if "order_purchase_timestamp" in df.columns and "order_delivered_customer_date" in df.columns:
            df["delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
        else:
            df["delivery_days"] = np.nan

        df = df.dropna(subset=["delivery_days"])
        if df.empty:
            return {
                "avg_current_delivery_days": None,
                "classification": {"fast": None, "medium": None, "slow": None},
                "percentiles": {}
            }

        # percentiles
        p25 = np.nanpercentile(df["delivery_days"], 25)
        p50 = np.nanpercentile(df["delivery_days"], 50)
        p75 = np.nanpercentile(df["delivery_days"], 75)

        def label_speed(x):
            if x <= p25:
                return "fast"
            if x <= p75:
                return "medium"
            return "slow"

        df["delivery_speed"] = df["delivery_days"].apply(label_speed)

        speed_dist = df["delivery_speed"].value_counts(normalize=True).to_dict()
        avg_delivery = float(round(df["delivery_days"].mean(),3))

        return {
            "avg_current_delivery_days": avg_delivery,
            "speed_distribution": speed_dist,
            "percentiles": {"p25": float(p25), "p50": float(p50), "p75": float(p75)}
        }

    def _analyze_economic_relations_and_trend(self):
        # Correlations global
        econ = self.df_economic.copy()
        correlations = {}
        cols = ["econ_act","peo_debt","inflation","interest_rate"]
        for col in cols:
            if col in econ.columns and econ[col].dtype in [np.float64, np.int64, np.int32, np.float32]:
                try:
                    correlations[col] = round(float(econ["econ_act"].corr(econ[col])),3)
                except Exception:
                    correlations[col] = None
            else:
                correlations[col] = None

        # Monthly orders series for trend
        orders = self.df_orders.copy()
        if "order_purchase_timestamp" in orders.columns:
            orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
            orders["year_month"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
            monthly = orders.groupby("year_month").size().reset_index(name="orders_count")
            # join with econ by year_month if econ has that column
            if "date" in self.df_economic.columns:
                econ = self.df_economic.copy()
                # try to standardize econ.date to year-month
                try:
                    econ["year_month"] = pd.to_datetime(econ["date"], errors="coerce").dt.to_period("M").astype(str)
                    joined = monthly.merge(econ, on="year_month", how="left")
                except Exception:
                    joined = monthly
            else:
                joined = monthly

            # trend via simple linear regression of month index
            joined = joined.dropna(subset=["orders_count"])
            if len(joined) >= 3:
                joined = joined.reset_index(drop=True)
                joined["month_idx"] = np.arange(len(joined))
                X = joined[["month_idx"]].values
                y = joined["orders_count"].values
                lr = LinearRegression()
                try:
                    lr.fit(X,y)
                    slope = float(lr.coef_[0])
                    trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                except Exception:
                    slope = None
                    trend = None
            else:
                slope = None
                trend = None
        else:
            joined = pd.DataFrame()
            slope = None
            trend = None

        return {
            "national_correlations": correlations,
            "monthly_orders_joined": joined.to_dict(orient="records") if not joined.empty else [],
            "trend_estimate": {"slope": slope, "trend": trend}
        }
