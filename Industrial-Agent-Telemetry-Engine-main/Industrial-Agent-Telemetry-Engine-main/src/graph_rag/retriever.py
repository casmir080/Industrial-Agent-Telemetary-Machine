import os
import warnings
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Suppress benign warnings for a clean production CLI
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*Qdrant client version.*")

class HybridRetriever:
    """Executes simultaneous queries against Neo4j (Topology) and Qdrant (Semantics)."""
    
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.qdrant = QdrantClient("http://localhost:6333")
        self.collection_name = "telemetry_knowledge"
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "telemetry_admin_123"))

    def search_qdrant(self, query: str, top_k: int = 2) -> list:
        """Semantic search for unstructured maintenance notes."""
        vector = self.encoder.encode(query).tolist()
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=top_k
        )
        return [hit.payload["text"] for hit in results]

    def search_neo4j(self, component: str) -> list:
        """Topological search for explicit component-procedure relationships."""
        query = """
        MATCH (c:Component)-[:REQUIRES_ACTION]->(p:Procedure)
        WHERE c.name CONTAINS $component
        RETURN c.name AS component, p.description AS procedure
        """
        procedures = []
        with self.neo4j_driver.session() as session:
            result = session.run(query, component=component)
            for record in result:
                procedures.append(f"Requires: {record['procedure']} (Component: {record['component']})")
        return procedures

    def hybrid_search(self, natural_query: str, target_component: str) -> dict:
        """Fuses vector semantics with graph topology."""
        semantic_results = self.search_qdrant(natural_query)
        topological_results = self.search_neo4j(target_component)
        return {
            "vector_context": semantic_results,
            "graph_context": topological_results
        }

if __name__ == "__main__":
    print("Initializing Hybrid Retriever Verification...")
    try:
        retriever = HybridRetriever()
        test_query = "How to handle severe thermal degradation?"
        target_component = "High Pressure Compressor"
        results = retriever.hybrid_search(natural_query=test_query, target_component=target_component)
        print(f"[SUCCESS] Hybrid Retrieval executed cleanly.")
        print("\n[METRICS] Vector Context Retrieved (Qdrant):")
        for idx, text in enumerate(results['vector_context']):
            print(f"  {idx + 1}. {text.strip()[:80]}...")
        print("\n[METRICS] Graph Context Retrieved (Neo4j):")
        for idx, text in enumerate(results['graph_context']):
            print(f"  {idx + 1}. {text}")
    except Exception as e:
        print(f"[ERROR] {str(e)}")