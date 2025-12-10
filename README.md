# Sistema ETL de Análisis de E-commerce Brasil

## 📋 Descripción General

Este proyecto es un sistema completo de procesamiento ETL (Extract, Transform, Load) diseñado para analizar datos de e-commerce brasileño utilizando el dataset público de Olist. El sistema procesa información de pedidos, clientes, productos, geolocalización e indicadores económicos para determinar ubicaciones óptimas de almacenes mediante técnicas de clustering y análisis avanzado.

## 🏗️ Arquitectura del Proyecto

El proyecto está estructurado en dos componentes principales:

### Backend (Python)
- **Ubicación**: `backend/procesamiento/`
- **Propósito**: Procesamiento ETL, análisis de datos y carga en MongoDB
- **Tecnologías**: Python 3.x, Pandas, NumPy, Scikit-learn, PyMongo

### Frontend (React)
- **Ubicación**: `frontend/`
- **Propósito**: Visualización de datos y resultados
- **Tecnologías**: React 19, Vite, ECharts

## 📁 Estructura de Directorios

```
.
├── backend/
│   └── procesamiento/
│       ├── data/                          # Datasets CSV
│       │   ├── olist_orders_dataset.csv
│       │   ├── olist_customers_dataset.csv
│       │   ├── olist_order_items_dataset.csv
│       │   ├── olist_products_dataset.csv
│       │   ├── olist_sellers_dataset.csv
│       │   ├── olist_geolocation_dataset.csv
│       │   └── brazil_economy_indicators.csv
│       ├── etl/
│       │   ├── config.py                  # Configuración de rutas y MongoDB
│       │   ├── database/
│       │   │   ├── mongo_handler.py       # Manejo de conexión MongoDB
│       │   │   └── create_economic_collection.py
│       │   └── processing/
│       │       ├── data_cleaner.py         # Limpieza y carga de datos
│       │       ├── data_processor.py       # Orquestador principal ETL
│       │       ├── metric_calculator.py    # Cálculo de métricas
│       │       ├── warehouse_allocator.py  # Clustering y ubicación de almacenes
│       │       ├── economic_analyzer.py   # Análisis económico
│       │       └── delivery_analyzer.py   # Análisis de entregas
│       ├── main.py                         # Punto de entrada
│       ├── requirements.txt                # Dependencias Python
│       └── test_connection.py             # Script de prueba MongoDB
└── frontend/
    ├── src/
    │   ├── App.jsx                         # Componente principal
    │   ├── HistoricPage.jsx               # Página histórica
    │   └── main.jsx                        # Entry point React
    ├── package.json                        # Dependencias Node.js
    └── vite.config.js                      # Configuración Vite
```

## 🔄 Flujo de Procesamiento ETL

### Fase 1: Extracción y Limpieza (Extract & Transform)

El proceso comienza en `main.py` y sigue estos pasos:

1. **Carga de Datasets** (`DataCleaner.load_all_datasets()`)
   - Carga 7 archivos CSV desde `backend/procesamiento/data/`
   - Datasets: órdenes, clientes, items, productos, vendedores, geolocalización, indicadores económicos

2. **Filtrado de Datos** (`DataCleaner.filter_delivered_orders()`)
   - Filtra solo órdenes con estado "delivered"
   - Elimina datos incompletos o inválidos

3. **Limpieza de Datos** (`DataCleaner.clean_datasets()`)
   - Normalización de formatos de fecha
   - Manejo de valores nulos
   - Validación de tipos de datos

### Fase 2: Procesamiento y Análisis (Transform)

El `DataProcessor` ejecuta múltiples análisis:

#### 2.1 Cálculo de Métricas (`MetricCalculator`)

- **Métricas Generales**:
  - Total de clientes únicos
  - Total de items vendidos
  - Promedio de items por cliente

- **Análisis de Entregas** (`DeliveryAnalyzer`):
  - Cálculo de días de entrega
  - Clasificación: rápida, media, lenta (percentiles 25, 50, 75)
  - Estadísticas por estado brasileño
  - Tendencias temporales de velocidad de entrega

- **Análisis Económico** (`EconomicAnalyzer`):
  - Correlación entre volumen de pedidos e indicadores económicos:
    - Actividad económica (`econ_act`)
    - Deuda pública (`peo_debt`)
    - Inflación (`inflation`)
    - Tasa de interés (`interest_rate`)
  - Volúmenes mensuales de pedidos
  - Tendencias de crecimiento/decrecimiento

#### 2.2 Clustering para Ubicación de Almacenes (`WarehouseAllocator`)

El sistema utiliza **tres algoritmos de clustering** para determinar ubicaciones óptimas:

1. **KMeans** (Clásico)
   - Selección automática de clusters mediante método del codo
   - Rango de búsqueda: 5 a máximo adaptativo (sqrt(n_puntos) / 2)

2. **MiniBatchKMeans** (Optimizado para grandes datasets)
   - Mismo método de selección que KMeans
   - Batch size: 2048
   - Más eficiente en memoria

3. **GMM (Gaussian Mixture Model)**
   - Selección mediante criterio BIC (Bayesian Information Criterion)
   - Rango adaptativo: mínimo 25, máximo 75 clusters

**Proceso de Clustering**:

1. **Preparación de Coordenadas**:
   - Merge de clientes con geolocalización por código postal
   - Extracción de coordenadas lat/lng válidas

2. **Clustering Principal**:
   - Aplicación del algoritmo seleccionado
   - Asignación de cada cliente a un cluster

3. **Cálculo de Centros de Almacenes**:
   - Para cada cluster:
     - Cálculo del centroide geográfico
     - Eliminación de outliers (percentil 95)
     - Cálculo de densidad de clientes
     - Identificación de productos más vendidos

4. **Subclustering Automático**:
   - Si un cluster tiene densidad > 8% del total:
     - Se divide automáticamente en subclusters (máximo 3)
     - Permite mayor granularidad en zonas de alta densidad

5. **Clasificación de Tamaño**:
   - **Large**: densidad > 4% del total
   - **Medium**: densidad entre 1.5% y 4%
   - **Small**: densidad < 1.5%

6. **Estimación de Mejora de Entrega**:
   - Cálculo de mejora porcentual estimada (10% - 25%)
   - Basado en la densidad del cluster

#### 2.3 Proyección de Crecimiento

Para cada almacén, se calcula crecimiento proyectado a 1 y 2 años:

```
growth_factor = 0.5 * econ_act - 0.2 * peo_debt - 0.1 * inflation - 0.1 * interest_rate
growth_factor = clamp(growth_factor, -0.5, 1.0)

estimated_customers_1y = current_customers * (1 + growth_factor)
estimated_customers_2y = current_customers * (1 + growth_factor)²
```

### Fase 3: Carga en MongoDB (Load)

El `MongoDBHandler` gestiona la persistencia:

1. **Conexión**:
   - URI de MongoDB desde variable de entorno `MONGODB_URI`
   - Base de datos: `ecommerce_brazil` (configurable)

2. **Colecciones Creadas**:
   - `orders`: Órdenes procesadas
   - `order_items`: Items de cada orden
   - `customers`: Información de clientes
   - `sellers`: Información de vendedores
   - `products`: Catálogo de productos
   - `geolocation`: Datos geográficos
   - `economic_data`: Indicadores económicos
   - `processed_results_kmeans`: Resultados del modelo KMeans
   - `processed_results_minibatch`: Resultados del modelo MiniBatchKMeans
   - `processed_results_gmm`: Resultados del modelo GMM

3. **Estructura de `processed_results_*`**:
```json
{
  "timestamp": "ISO 8601",
  "metrics": {
    "total_customers": int,
    "total_items": int,
    "items_per_customer_avg": float,
    "total_warehouses": int,
    "avg_customers_per_warehouse": int
  },
  "economic_analysis": {
    "national_correlations": {...},
    "monthly_volumes": [...],
    "trend_estimates": {...}
  },
  "delivery_stats": {
    "summary": {...},
    "by_state": [...]
  },
  "warehouses": [
    {
      "warehouse_id": str/int,
      "latitude": float,
      "longitude": float,
      "customer_count": int,
      "density_ratio": float,
      "warehouse_size": "large|medium|small",
      "estimated_delivery_improvement_%": float,
      "top_items": [str],
      "estimated_customer_growth_1y": int,
      "estimated_customer_growth_2y": int,
      "note": str,
      "algorithm": "kmeans|minibatch|gmm"
    }
  ],
  "cluster_logs": [
    {
      "algorithm": str,
      "total_warehouses": int,
      "large": int,
      "medium": int,
      "small": int
    }
  ],
  "notes": {
    "clustering_method": str,
    "n_clusters": int
  }
}
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- Node.js 16+ (para frontend)
- MongoDB Atlas (o MongoDB local)
- Cuenta de MongoDB Atlas con cluster creado

### Backend

1. **Instalar dependencias**:
```bash
cd backend/procesamiento
pip install -r requirements.txt
```

2. **Configurar variables de entorno**:
Crear archivo `.env` en `backend/procesamiento/`:
```
MONGODB_URI=
MONGODB_DATABASE=
```

3. **Ejecutar el proceso ETL**:
```bash
python main.py
```

### Frontend

1. **Instalar dependencias**:
```bash
cd frontend
npm install
```

2. **Ejecutar en modo desarrollo**:
```bash
npm run dev
```

3. **Compilar para producción**:
```bash
npm run build
```

## 📊 Datasets Utilizados

### Datasets Olist
- **orders**: Información de pedidos (estado, fechas, cliente)
- **order_items**: Items individuales de cada pedido
- **customers**: Datos demográficos de clientes
- **sellers**: Información de vendedores
- **products**: Catálogo de productos
- **geolocation**: Coordenadas geográficas por código postal

### Dataset Económico
- **brazil_economy_indicators**: Indicadores macroeconómicos mensuales de Brasil

## 🔍 Algoritmos de Clustering

### KMeans
- **Ventaja**: Rápido y eficiente
- **Uso**: Dataset estándar
- **Selección de K**: Método del codo con segunda derivada

### MiniBatchKMeans
- **Ventaja**: Optimizado para grandes volúmenes de datos
- **Uso**: Cuando el dataset es muy grande
- **Batch Size**: 2048

### GMM (Gaussian Mixture Model)
- **Ventaja**: Modela distribuciones probabilísticas más complejas
- **Uso**: Cuando se esperan clusters con formas no esféricas
- **Selección de componentes**: Criterio BIC

## 📈 Métricas y Análisis Generados

### Métricas de Negocio
- Total de clientes únicos
- Total de items vendidos
- Promedio de items por cliente
- Total de almacenes recomendados
- Promedio de clientes por almacén

### Análisis de Entregas
- Distribución de velocidad (rápida/media/lenta)
- Tiempo promedio de entrega por estado
- Tendencias temporales

### Análisis Económico
- Correlaciones entre pedidos e indicadores económicos
- Volúmenes mensuales históricos
- Tendencias de crecimiento/decrecimiento

### Análisis Geográfico
- Ubicaciones óptimas de almacenes (lat/lng)
- Densidad de clientes por región
- Productos más vendidos por región
- Proyección de crecimiento de clientes

## 🛠️ Módulos Principales

### `DataCleaner`
Responsable de:
- Carga de archivos CSV
- Filtrado de datos relevantes
- Limpieza y normalización

### `DataProcessor`
Orquestador principal que:
- Coordina el flujo ETL completo
- Ejecuta los tres modelos de clustering
- Genera resultados estructurados

### `MetricCalculator`
Calcula:
- Métricas generales del negocio
- Estadísticas de entregas
- Análisis económico y correlaciones

### `WarehouseAllocator`
Implementa:
- Tres algoritmos de clustering
- Selección automática de número de clusters
- Cálculo de ubicaciones óptimas
- Clasificación por tamaño

### `EconomicAnalyzer`
Analiza:
- Correlaciones económicas
- Tendencias temporales
- Volúmenes mensuales

### `DeliveryAnalyzer`
Evalúa:
- Performance de entregas
- Clasificación por velocidad
- Estadísticas por estado

### `MongoDBHandler`
Gestiona:
- Conexión a MongoDB
- Inserción de colecciones
- Limpieza de colecciones existentes

## 🔧 Configuración Avanzada

### Ajustar Número de Clusters

En `main.py`, puedes especificar el número de clusters:
```python
processor = DataProcessor()
processor.execute_etl(n_clusters=20)  # Fuerza 20 clusters
```

Si no se especifica, el sistema selecciona automáticamente el óptimo.

### Modificar Parámetros de Clustering

Editar `warehouse_allocator.py`:
- `max_clusters`: Límite superior para búsqueda
- `batch_size`: Tamaño de batch para MiniBatchKMeans
- Umbrales de densidad para clasificación de tamaños

### Personalizar Proyección de Crecimiento

En `data_processor.py`, función `apply_growth()`:
```python
growth_factor = 0.5 * norm_econ_act - 0.2 * norm_peo_debt - 0.1 * norm_inflation - 0.1 * norm_interest_rate
```
Ajustar los coeficientes según necesidades del negocio.

