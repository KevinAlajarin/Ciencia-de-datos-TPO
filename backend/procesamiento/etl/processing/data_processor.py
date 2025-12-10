import pandas as pd
from datetime import datetime
from .data_cleaner import DataCleaner
from .metric_calculator import MetricCalculator
from .warehouse_allocator import WarehouseAllocator

class DataProcessor:
    """
    Orquesta el proceso ETL completo con mejoras de clustering, métricas y proyección de crecimiento de clientes.
    Ejecuta los tres modelos de clustering (KMeans, MiniBatchKMeans, GMM) y genera tres documentos:
      - processed_results_kmeans
      - processed_results_minibatch
      - processed_results_gmm

    Además mantiene `processed_results` apuntando a KMeans para compatibilidad con código previo.
    """

    def __init__(self, cleaner=None):
        self.cleaner = cleaner if cleaner else DataCleaner()
        self.calculator = None
        # resultados por modelo
        self.processed_results_kmeans = {}
        self.processed_results_minibatch = {}
        self.processed_results_gmm = {}
        # compatibilidad hacia atrás (apunta a kmeans por defecto)
        self.processed_results = {}

    def execute_etl(self, n_clusters=None):
        print("Iniciando proceso ETL completo...")

        if not self.cleaner.load_all_datasets():
            print("Error cargando datasets.")
            return False

        # Filtrar y limpiar
        self.cleaner.filter_delivered_orders()
        self.cleaner.clean_datasets()

        # Instanciar metric calculator
        try:
            self.calculator = MetricCalculator(
                df_orders=self.cleaner.datasets.get("orders"),
                df_items=self.cleaner.datasets.get("order_items"),
                df_customers=self.cleaner.datasets.get("customers"),
                df_geolocation=self.cleaner.datasets.get("geolocation"),
                df_economic=self.cleaner.datasets.get("economic_indicators"),
                df_products=self.cleaner.datasets.get("products")
            )
        except Exception as e:
            print(f"Error al instanciar MetricCalculator: {e}")
            return False

        print("Calculando métricas y ubicaciones de warehouses...")

        try:
            # 1) Calcular métricas generales y económicas
            results_base = self.calculator.calculate_all()

            
            # 2) Ejecutar modelos de clustering
            
            # KMEANS (comportamiento original)
            allocator_k = WarehouseAllocator(
                df_orders=self.cleaner.datasets.get("orders"),
                df_customers=self.cleaner.datasets.get("customers"),
                df_geolocation=self.cleaner.datasets.get("geolocation"),
                df_items=self.cleaner.datasets.get("order_items"),
                df_products=self.cleaner.datasets.get("products"),
                n_clusters=n_clusters
            )
            warehouses_k = allocator_k.estimate(algorithm="kmeans")

            # MINIBATCH
            allocator_mb = WarehouseAllocator(
                df_orders=self.cleaner.datasets.get("orders"),
                df_customers=self.cleaner.datasets.get("customers"),
                df_geolocation=self.cleaner.datasets.get("geolocation"),
                df_items=self.cleaner.datasets.get("order_items"),
                df_products=self.cleaner.datasets.get("products"),
                n_clusters=n_clusters
            )
            warehouses_mb = allocator_mb.estimate(algorithm="minibatch")

            # GMM
            allocator_gmm = WarehouseAllocator(
                df_orders=self.cleaner.datasets.get("orders"),
                df_customers=self.cleaner.datasets.get("customers"),
                df_geolocation=self.cleaner.datasets.get("geolocation"),
                df_items=self.cleaner.datasets.get("order_items"),
                df_products=self.cleaner.datasets.get("products"),
                n_clusters=n_clusters
            )
            warehouses_gmm = allocator_gmm.estimate(algorithm="gmm")

            
            # 3) Proyección de crecimiento por warehouse (1 y 2 años)
            #    (misma regla aplicada a los 3 listados)
            
            econ = results_base.get("economic_analysis", {}).get("national_correlations", {})
            econ_act = econ.get("econ_act", 0.0)
            peo_debt = econ.get("peo_debt", 0.0)
            inflation = econ.get("inflation", 0.0)
            interest_rate = econ.get("interest_rate", 0.0)

            # Normalización simple 0-1
            norm_econ_act = min(max(econ_act, 0), 1)
            norm_peo_debt = min(max(peo_debt, 0), 1)
            norm_inflation = min(max(inflation, 0), 1)
            norm_interest_rate = min(max(interest_rate, 0), 1)

            def apply_growth(warehouses_list):
                for w in warehouses_list:
                    growth_factor = 0.5 * norm_econ_act - 0.2 * norm_peo_debt - 0.1 * norm_inflation - 0.1 * norm_interest_rate
                    # proteger / acotar factor
                    growth_factor = max(-0.5, min(1.0, growth_factor))
                    # si no existe customer_count proteger
                    cust = w.get("customer_count", 0) or 0
                    w["estimated_customer_growth_1y"] = int(cust * (1 + growth_factor))
                    w["estimated_customer_growth_2y"] = int(cust * (1 + growth_factor) ** 2)
                return warehouses_list

            warehouses_k = apply_growth(warehouses_k)
            warehouses_mb = apply_growth(warehouses_mb)
            warehouses_gmm = apply_growth(warehouses_gmm)

            
            # 4) Construir processed_results por cada modelo
            
            base_metrics = results_base.get("metrics", {}).copy()

            # KMEANS
            processed_k = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": base_metrics.copy(),
                "economic_analysis": results_base.get("economic_analysis", {}),
                "delivery_stats": results_base.get("delivery_stats", {}),
                "warehouses": warehouses_k,
                "cluster_logs": allocator_k.logs,
                "notes": {
                    "clustering_method": "KMeans",
                    "n_clusters": allocator_k.n_clusters
                }
            }
            total_wh_k = len(warehouses_k)
            processed_k["metrics"]["total_warehouses"] = total_wh_k
            if total_wh_k > 0 and processed_k["metrics"].get("total_customers"):
                processed_k["metrics"]["avg_customers_per_warehouse"] = int(processed_k["metrics"]["total_customers"] / total_wh_k)

            # MINIBATCH
            processed_mb = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": base_metrics.copy(),
                "economic_analysis": results_base.get("economic_analysis", {}),
                "delivery_stats": results_base.get("delivery_stats", {}),
                "warehouses": warehouses_mb,
                "cluster_logs": allocator_mb.logs,
                "notes": {
                    "clustering_method": "MiniBatchKMeans",
                    "n_clusters": allocator_mb.n_clusters
                }
            }
            total_wh_mb = len(warehouses_mb)
            processed_mb["metrics"]["total_warehouses"] = total_wh_mb
            if total_wh_mb > 0 and processed_mb["metrics"].get("total_customers"):
                processed_mb["metrics"]["avg_customers_per_warehouse"] = int(processed_mb["metrics"]["total_customers"] / total_wh_mb)

            # GMM
            processed_gmm = {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": base_metrics.copy(),
                "economic_analysis": results_base.get("economic_analysis", {}),
                "delivery_stats": results_base.get("delivery_stats", {}),
                "warehouses": warehouses_gmm,
                "cluster_logs": allocator_gmm.logs,
                "notes": {
                    "clustering_method": "GMM",
                    "n_clusters": allocator_gmm.n_clusters
                }
            }
            total_wh_gmm = len(warehouses_gmm)
            processed_gmm["metrics"]["total_warehouses"] = total_wh_gmm
            if total_wh_gmm > 0 and processed_gmm["metrics"].get("total_customers"):
                processed_gmm["metrics"]["avg_customers_per_warehouse"] = int(processed_gmm["metrics"]["total_customers"] / total_wh_gmm)

            
            # 5) Guardar resultados en atributos y compatibilidad
            
            self.processed_results_kmeans = processed_k
            self.processed_results_minibatch = processed_mb
            self.processed_results_gmm = processed_gmm

            # compatibilidad: processed_results es el documento KMeans por defecto
            self.processed_results = self.processed_results_kmeans

            print("Proceso ETL completado correctamente (KMeans + MiniBatch + GMM).")
            return True

        except Exception as e:
            print(f"Error en el cálculo de métricas o ubicación: {e}")
            return False

    def get_processed_data(self):
        return {
            "datasets_originales": self.cleaner.get_all_datasets(),
            "processed_results_kmeans": self.processed_results_kmeans,
            "processed_results_minibatch": self.processed_results_minibatch,
            "processed_results_gmm": self.processed_results_gmm
        }

    def prepare_mongodb_documents(self):
        """
        Prepara el dict de colecciones -> lista de documentos para insertar en MongoDB.
        Mantiene las colecciones originales y añade:
          - processed_results_kmeans
          - processed_results_minibatch
          - processed_results_gmm
        """
        datasets = self.cleaner.get_all_datasets()
        mongo_docs = {name: df.to_dict("records") for name, df in datasets.items()}

        # push both processed results as single-document lists (si existen)
        mongo_docs["processed_results_kmeans"] = [self.processed_results_kmeans] if self.processed_results_kmeans else []
        mongo_docs["processed_results_minibatch"] = [self.processed_results_minibatch] if self.processed_results_minibatch else []
        mongo_docs["processed_results_gmm"] = [self.processed_results_gmm] if self.processed_results_gmm else []

        return mongo_docs
