import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist


class WarehouseAllocator:
    """
    Asigna ubicaciones óptimas de warehouse usando clustering geográfico.
    Mantiene toda la lógica original y añade:
    - Selección automática de clusters para KMeans
    - Selección automática de clusters para MiniBatchKMeans
    - Selección automática para GMM mediante BIC
    - Log de resultados de warehouses por tamaño
    """

    def __init__(self, df_orders, df_customers, df_geolocation, df_items, df_products, n_clusters=None):
        self.df_orders = df_orders
        self.df_customers = df_customers
        self.df_geolocation = df_geolocation
        self.df_items = df_items
        self.df_products = df_products
        self.n_clusters = n_clusters
        self.logs = []

    # Selección automática de clusters KMeans / MiniBatch
    def _select_best_kmeans(self, coords, minibatch=False, max_clusters=None):
        n_points = len(coords)
        if max_clusters is None:
            max_clusters = max(10, min(50, int(np.sqrt(n_points) // 2)))

        print(f"Buscando cantidad óptima de clusters (KMeans/MiniBatch) hasta k={max_clusters}...")

        inertias = []
        models = []

        K_range = range(5, max_clusters + 1)

        for k in K_range:
            try:
                if minibatch:
                    model = MiniBatchKMeans(
                        n_clusters=k,
                        random_state=42,
                        batch_size=2048,
                        n_init="auto"
                    )
                else:
                    model = KMeans(
                        n_clusters=k,
                        random_state=42,
                        n_init=10
                    )

                model.fit(coords)
                inertias.append(model.inertia_)
                models.append(model)

            except Exception:
                break

        reductions = np.diff(inertias)
        second_diff = np.diff(reductions)
        elbow_index = np.argmin(np.abs(second_diff)) + 2
        best_k = list(K_range)[elbow_index]

        print(f"Cluster óptimo encontrado: k = {best_k}")
        return models[elbow_index], best_k

    # Selección automática de clusters para GMM mediante BIC
    def _select_best_gmm(self, coords, max_clusters=None):
        """
        Selecciona automáticamente el mejor número de clusters usando GaussianMixture y BIC.
        max_clusters: límite superior adaptativo según tamaño del dataset.
        """
        n_points = len(coords)

        # Regla heurística: sqrt(n_points), mínimo 25, máximo 75 para no exagerar
        if max_clusters is None:
            max_clusters = int(np.sqrt(n_points))
            max_clusters = max(25, min(max_clusters, 75))

        print(f"Buscando cantidad óptima de clusters (GMM) hasta k={max_clusters}...")

        lowest_bic = np.inf
        best_gmm = None

        for k in range(1, max_clusters + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    random_state=42
                )
                gmm.fit(coords)
                bic = gmm.bic(coords)

                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm

            except Exception:
                continue

        print(f"GMM óptimo encontrado: n_components = {best_gmm.n_components}")
        return best_gmm

    # Estimación general
    def estimate(self, algorithm="kmeans"):
        print(f"Estimando warehouses | algoritmo={algorithm}")

        df_cust = self.df_customers.copy()
        df_geo = self.df_geolocation.copy()

        if "customer_zip_code_prefix" not in df_cust.columns:
            zip_col = [c for c in df_cust.columns if "zip" in c][0]
            df_cust = df_cust.rename(columns={zip_col: "customer_zip_code_prefix"})

        if "geolocation_zip_code_prefix" not in df_geo.columns:
            zip_col = [c for c in df_geo.columns if "zip" in c][0]
            df_geo = df_geo.rename(columns={zip_col: "geolocation_zip_code_prefix"})

        df_merge = pd.merge(
            df_cust,
            df_geo,
            left_on="customer_zip_code_prefix",
            right_on="geolocation_zip_code_prefix",
            how="left"
        )

        df_merge["geolocation_lat"] = pd.to_numeric(df_merge["geolocation_lat"], errors="coerce")
        df_merge["geolocation_lng"] = pd.to_numeric(df_merge["geolocation_lng"], errors="coerce")
        df_merge = df_merge.dropna(subset=["geolocation_lat", "geolocation_lng"])

        if df_merge.empty:
            raise ValueError("No coordenadas válidas")

        coords = df_merge[["geolocation_lat", "geolocation_lng"]].values

        if algorithm == "minibatch":
            model, optimal_k = self._select_best_kmeans(coords, minibatch=True)
            df_merge["cluster"] = model.predict(coords)
            self.n_clusters = optimal_k

        elif algorithm == "gmm":
            best_gmm = self._select_best_gmm(coords)  # usa límite dinámico
            df_merge["cluster"] = best_gmm.predict(coords)
            self.n_clusters = best_gmm.n_components

        else:  # kmeans default
            model, optimal_k = self._select_best_kmeans(coords, minibatch=False)
            df_merge["cluster"] = model.predict(coords)
            self.n_clusters = optimal_k

        # Vinculación con items y productos
        df_full = (
            self.df_orders.merge(self.df_items, on="order_id", how="inner")
            .merge(self.df_products, on="product_id", how="left")
            .merge(df_merge[["customer_id", "cluster"]], on="customer_id", how="left")
        )

        warehouses = []
        valid_clusters = sorted(df_full["cluster"].dropna().unique())
        total_customers = df_merge["customer_id"].nunique()

        for cluster_id in valid_clusters:
            cluster_points = df_merge[df_merge["cluster"] == cluster_id]
            if cluster_points.empty:
                continue

            lat_mean = cluster_points["geolocation_lat"].mean()
            lon_mean = cluster_points["geolocation_lng"].mean()
            coords_cluster = cluster_points[["geolocation_lat", "geolocation_lng"]].values
            centroid = np.array([lat_mean, lon_mean])
            distances = cdist(coords_cluster, [centroid])
            outlier_mask = distances[:, 0] > np.percentile(distances, 95)
            cluster_points_clean = cluster_points[~outlier_mask]

            density = cluster_points_clean["customer_id"].nunique()
            ratio = density / total_customers

            cluster_items = df_full[df_full["cluster"] == cluster_id]
            top_items = cluster_items["product_id"].value_counts().head(5).index.tolist()

            # subclusters automáticos
            if ratio > 0.08:
                sub_k = min(3, int(ratio * 100))
                sub_model = KMeans(n_clusters=sub_k, random_state=42)
                sub_labels = sub_model.fit_predict(
                    cluster_points_clean[["geolocation_lat", "geolocation_lng"]].values
                )

                for sub_id in range(sub_k):
                    sub_pts = cluster_points_clean[sub_labels == sub_id]
                    if sub_pts.empty:
                        continue

                    sub_lat = sub_pts["geolocation_lat"].mean()
                    sub_lon = sub_pts["geolocation_lng"].mean()
                    sub_density = sub_pts["customer_id"].nunique()
                    sub_ratio = sub_density / total_customers

                    size = (
                        "large" if sub_ratio > 0.04 else
                        "medium" if sub_ratio > 0.015 else
                        "small"
                    )

                    base, maxv = 10, 25
                    improvement = round(base + (maxv - base) * min(sub_ratio / 0.1, 1), 2)

                    warehouses.append({
                        "warehouse_id": f"{cluster_id}_{sub_id}",
                        "latitude": float(sub_lat),
                        "longitude": float(sub_lon),
                        "customer_count": int(sub_density),
                        "density_ratio": round(sub_ratio, 4),
                        "warehouse_size": size,
                        "estimated_delivery_improvement_%": improvement,
                        "top_items": top_items,
                        "note": "Subcluster automático",
                        "algorithm": algorithm
                    })

                continue

            size = (
                "large" if ratio > 0.04 else
                "medium" if ratio > 0.015 else
                "small"
            )
            base, maxv = 10, 25
            improvement = round(base + (maxv - base) * min(ratio / 0.1, 1), 2)

            warehouses.append({
                "warehouse_id": int(cluster_id),
                "latitude": float(lat_mean),
                "longitude": float(lon_mean),
                "customer_count": int(density),
                "density_ratio": round(ratio, 4),
                "warehouse_size": size,
                "estimated_delivery_improvement_%": improvement,
                "top_items": top_items,
                "note": "Cluster normal",
                "algorithm": algorithm
            })

        # Log automático por tamaños
        sizes = [wh["warehouse_size"] for wh in warehouses]
        log_summary = {
            "algorithm": algorithm,
            "total_warehouses": len(warehouses),
            "large": sizes.count("large"),
            "medium": sizes.count("medium"),
            "small": sizes.count("small")
        }

        self.logs.append(log_summary)
        print(f"{algorithm.upper()} | Warehouses: {log_summary}")

        return warehouses
