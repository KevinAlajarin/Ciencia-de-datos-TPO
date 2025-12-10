from pymongo import MongoClient


class MongoDBHandler:
    """
    Clase manejadora de conexión y carga de datos a MongoDB.
    """

    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None


    # CONEXIÓN
    def connect(self):
        """
        Conecta al cluster de MongoDB Atlas y selecciona la base de datos.
        """
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            print("Conexión exitosa a MongoDB!")
            return True
        except Exception as e:
            print(f"Error conectando a MongoDB: {e}")
            return False


    # INSERCIÓN DE DATOS

    def insert_many(self, collection_name, documents):
        """
        Inserta una lista de documentos en una colección.
        Si la colección ya exista, la reemplaza completamente.
        """
        if self.db is None:
            raise Exception("Base de datos no inicializada. Llamar a connect() primero.")

        try:
            collection = self.db[collection_name]

            # Limpiar colección si ya existe
            if collection.estimated_document_count() > 0:
                collection.drop()

            if documents and len(documents) > 0:
                collection.insert_many(documents)
                print(f"Colección '{collection_name}' cargada con {len(documents)} documentos.")
            else:
                print(f"No hay documentos para insertar en '{collection_name}'.")

        except Exception as e:
            print(f"Error insertando en {collection_name}: {e}")
