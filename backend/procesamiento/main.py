import os
from dotenv import load_dotenv
from etl.processing.data_cleaner import DataCleaner
from etl.processing.data_processor import DataProcessor
from etl.database.mongo_handler import MongoDBHandler

def main():
    print("INICIANDO SISTEMA ETL - ECOMMERCE BRAZIL")
    print("==================================================\n")

    # FASE 1: PROCESAMIENTO ETL
    print("FASE 1: PROCESAMIENTO ETL")
    load_dotenv()

    processor = DataProcessor()

    if not processor.execute_etl():
        print("Error durante la fase ETL.")
        return
    print("ETL completado exitosamente\n")

    # FASE 2: CARGA EN MONGODB
    print("FASE 2: CARGA EN MONGODB")

    mongo_uri = os.getenv("MONGODB_URI")
    mongo_db_name = os.getenv("MONGODB_DATABASE", "ecommerce_brazil")

    mongo_handler = MongoDBHandler(mongo_uri, mongo_db_name)

    if not mongo_handler.connect():
        print("Error al conectar con MongoDB.")
        return

    print("Subiendo colecciones a MongoDB...")

    try:
        mongo_docs = processor.prepare_mongodb_documents()
        # insertar todas las colecciones preparadas
        for name, records in mongo_docs.items():
            mongo_handler.insert_many(name, records)
        print("Datos cargados exitosamente en MongoDB\n")

    except Exception as e:
        print(f"Error al cargar datos en MongoDB: {e}")
        return

    # FASE 3: ANÁLISIS Y RESULTADOS (compatibilidad: processed_results apunta a KMeans)
    print("FASE 3: ANÁLISIS Y RESULTADOS")
    results = processor.processed_results or {}

    print("Resultados procesados:")
    warehouses = results.get("warehouses", [])
    correlations = results.get("economic_correlations", {}) or results.get("economic_analysis", {})
    metrics = results.get("metrics", {})


    print("\nSistema ETL finalizado correctamente")

if __name__ == "__main__":
    main()
