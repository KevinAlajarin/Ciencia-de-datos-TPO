# delivery_analyzer.py
import pandas as pd
import numpy as np

class DeliveryAnalyzer:
    def __init__(self, df_orders):
        self.df_orders = df_orders

    def analyze(self):
        df = self.df_orders.copy()
        df = df.dropna(subset=['order_purchase_timestamp', 'order_delivered_customer_date'])

        df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
        df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
        df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

        p25, p50, p75 = np.percentile(df['delivery_days'], [25, 50, 75])
        df['delivery_category'] = pd.cut(df['delivery_days'],
                                         bins=[-np.inf, p25, p75, np.inf],
                                         labels=['fast', 'medium', 'slow'])

        summary = df['delivery_category'].value_counts(normalize=True).to_dict()
        by_state = df.groupby('customer_state')['delivery_days'].agg(['mean', 'median']).reset_index()

        return {
            "summary": summary,
            "by_state": by_state.to_dict(orient='records')
        }
