# create_economic_collection.py
import pandas as pd
from .data_cleaner import DataCleaner
from .metric_calculator import MetricCalculator

class DataProcessor:
    """
    Clase principal de orquestación del proceso ETL:
    1. Carga y limpieza de datos
    2. Ejecución de cálculos analíticos y estimaciones
    3. Preparación de los resultados para MongoDB
    """

    def __init__(self):
        self.cleaner = DataCleaner()
        self.calculator = None
        self.processed_results = {}

    def execute_etl(self):
        """
        Ejecuta el pipeline ETL completo:
        - Carga datasets
        - Filtra pedidos entregados
        - Limpia datos
        - Calcula ubicación de warehouses, correlaciones económicas y métricas
        """
        print("Iniciando proceso ETL completo...")

        # CARGA
        if not self.cleaner.load_all_datasets():
            print("Error cargando datasets.")
            return False

        print("Todos los datasets cargados exitosamente")

        # FILTRADO
        self.cleaner.filter_delivered_orders()

        # LIMPIEZA
        print("Aplicando limpieza de datos...")
        self.cleaner.clean_datasets()
        print("Limpieza completada")

        # INSTANCIAR CALCULADORA
        try:
            self.calculator = MetricCalculator(
                df_items=self.cleaner.datasets["order_items"],
                df_customers=self.cleaner.datasets["customers"],
                df_geolocation=self.cleaner.datasets["geolocation"],
                df_economic=self.cleaner.datasets["economic_indicators"]
            )
        except Exception as e:
            print(f"Error al instanciar MetricCalculator: {e}")
            return False

        # CÁLCULOS
        print("Calculando ubicación del galpón y relación con datos económicos...")
        try:
            results = self.calculator.calculate_all()
            self.processed_results = results
            print("Cálculo económico-temporal completado.")
        except Exception as e:
            print(f"Error en el cálculo de métricas: {e}")
            return False

        print("Proceso ETL completado exitosamente.")
        return True

    def get_processed_data(self):
        """Retorna los datasets originales + resultados procesados."""
        return {
            "datasets_originales": self.cleaner.get_all_datasets(),
            "processed_results": self.processed_results,
        }

    def prepare_mongodb_documents(self):
        """Prepara documentos para carga en MongoDB."""
        datasets = self.cleaner.get_all_datasets()
        mongo_docs = {name: df.to_dict("records") for name, df in datasets.items()}
        mongo_docs["processed_results"] = [self.processed_results]
        return mongo_docs
