import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "data"

DATASET_FILES = {
    'customers': DATASETS_DIR / "olist_customers_dataset.csv",
    'geolocation': DATASETS_DIR / "olist_geolocation_dataset.csv",
    'order_items': DATASETS_DIR / "olist_order_items_dataset.csv",
    'orders': DATASETS_DIR / "olist_orders_dataset.csv",
    'products': DATASETS_DIR / "olist_products_dataset.csv",
    'sellers': DATASETS_DIR / "olist_sellers_dataset.csv",
    'economic_data': DATASETS_DIR / "brazil_economy_indicators.csv"
}

MONGODB_CONFIG = {
    'database_name': 'ecommerce_brazil',
    'collections': {
        'orders': 'orders',
        'order_items': 'order_items',
        'customers': 'customers',
        'sellers': 'sellers',
        'products': 'products',
        'geolocation': 'geolocation',
        'economic_data': 'economic_data',
        'processed_results': 'processed_results'
    }
}
