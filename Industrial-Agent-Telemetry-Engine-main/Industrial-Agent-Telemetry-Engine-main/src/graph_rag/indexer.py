import os
import uuid
import re
from pathlib import Path
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

class HybridKnowledgeIndexer:
    """End-to-end indexer mapping unstructured text to Qdrant (vectors) and Neo4j (graph topology)."""
    
    def __init__(self, knowledge_dir: str = "data/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize local embedding model
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_size = self.encoder.get_sentence_embedding_dimension()
        
        # Connect to Qdrant
        self.qdrant = QdrantClient("http://localhost:6333")
        self.collection_name = "telemetry_knowledge"
        self._setup_qdrant()
        
        # Connect to Neo4j
        self.neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "telemetry_admin_123"))
        self._setup_neo4j()
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    def _setup_qdrant(self):
        """Idempotent Qdrant collection creation."""
        if self.qdrant.collection_exists(self.collection_name):
            self.qdrant.delete_collection(self.collection_name)
        
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )

    def _setup_neo4j(self):
        """Idempotent Neo4j graph cleanup and constraint creation."""
        with self.neo4j_driver.session() as session:
            # Wipe existing graph for clean testing
            session.run("MATCH (n) DETACH DELETE n")
            # Enforce uniqueness on components
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE")

    def _ensure_sample_data(self):
        """Generates a sample technical log if the knowledge directory is empty."""
        sample_file = self.knowledge_dir / "maintenance_log_v1.txt"
        if not sample_file.exists():
            content = (
                "COMPONENT: High Pressure Compressor (HPC)\n"
                "OBSERVATION: Severe blade degradation and thermal scoring detected.\n"
                "PROCEDURE: Initiate turbine wash and replace HPC seal rings.\n\n"
                "COMPONENT: Fan Module\n"
                "OBSERVATION: Excessive vibration during takeoff spool-up.\n"
                "PROCEDURE: Rebalance fan blades and inspect bearing assemblies."
            )
            sample_file.write_text(content)
            return True
        return False

    def index_documents(self):
        self._ensure_sample_data()
        
        files = list(self.knowledge_dir.glob("*.txt"))
        total_chunks = 0
        total_nodes = 0
        
        for file_path in files:
            text = file_path.read_text()
            
            # 1. Regex-based Entity Extraction for Neo4j (Zero-Mock Rule: actual parsing)
            components = re.findall(r"COMPONENT:\s*(.+)", text)
            procedures = re.findall(r"PROCEDURE:\s*(.+)", text)
            
            with self.neo4j_driver.session() as session:
                for comp, proc in zip(components, procedures):
                    session.run(
                        """
                        MERGE (c:Component {name: $comp})
                        MERGE (p:Procedure {description: $proc})
                        MERGE (c)-[:REQUIRES_ACTION]->(p)
                        """,
                        comp=comp.strip(), proc=proc.strip()
                    )
                    total_nodes += 2 # Rough count of created nodes
            
            # 2. Text Chunking & Vector Encoding for Qdrant
            chunks = self.text_splitter.split_text(text)
            points = []
            
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                vector = self.encoder.encode(chunk).tolist()
                
                points.append(PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload={"source": file_path.name, "text": chunk}
                ))
            
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            total_chunks += len(points)
            
        return total_chunks, total_nodes

if __name__ == "__main__":
    print("Initializing Hybrid Graph-RAG Indexer...")
    try:
        indexer = HybridKnowledgeIndexer()
        chunks_indexed, nodes_created = indexer.index_documents()
        
        # Verification Queries
        # 1. Qdrant Verification
        qdrant_count = indexer.qdrant.count(indexer.collection_name).count
        
        # 2. Neo4j Verification
        with indexer.neo4j_driver.session() as session:
            neo4j_result = session.run("MATCH (c:Component)-[r:REQUIRES_ACTION]->(p:Procedure) RETURN count(r) as rel_count")
            rel_count = neo4j_result.single()["rel_count"]
            
        print(f"[SUCCESS] Hybrid Indexing completed cleanly.")
        print(f"[METRICS] Qdrant: {qdrant_count} vector chunks stored in '{indexer.collection_name}'")
        print(f"[METRICS] Neo4j: {rel_count} relationships (Component -> Procedure) established in graph.")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")