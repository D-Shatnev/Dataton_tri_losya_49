from qdrant_client import QdrantClient
import numpy as np

client = QdrantClient(host="localhost", port=6333)

def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm > 0:
        return (vec / norm).tolist()
    return vec


def add_vector(array, id_, collection):
    vector = normalize_vector(array)
    client.upsert(
        collection_name=collection,
        points=[
            {
                "id": id_,
                "vector": vector
            }
        ]
    )

def search_similar_vectors(query_vector, collection, k=10):
    """
    Поиск K ближайших векторов в коллекции Qdrant
    
    Args:
        client: QdrantClient instance
        collection_name: имя коллекции
        query_vector: вектор запроса (list или numpy array)
        k: количество ближайших соседей для возврата
    
    Returns:
        list of dict: список результатов с id и score
    """
    query_vector = normalize_vector(query_vector)
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()
    
    search_results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=k
    )
    
    results = [
        {
            "id": hit.id,
            "score": hit.score,
        }
        for hit in search_results.points
    ]
    
    return results