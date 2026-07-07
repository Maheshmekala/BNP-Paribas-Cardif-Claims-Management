"""
RAG (Retrieval-Augmented Generation) Pipeline for BNP Paribas Cardif Claims Management.
Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Enables semantic search over insurance policies, claim notes, and historical claims.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from backend.config import settings, logger


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for semantic search over claims data.

    Uses ChromaDB as the vector store with sentence-transformers for embeddings.
    Supports indexing of insurance policy documents, claim notes, and historical claims
    with few-shot examples stored for contextual retrieval.
    """

    def __init__(self):
        """Initialize the RAG pipeline with ChromaDB and embedding model."""
        self._chroma_available = False
        self._embedder_available = False
        self._embedding_model = None
        self._collections = {}
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check availability of optional dependencies."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            self._chroma_client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_PATH),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._chroma_available = True
            logger.info(f"ChromaDB initialized at {settings.VECTOR_DB_PATH}")
        except ImportError:
            self._chroma_client = None
            logger.warning("ChromaDB not available. Install with: pip install chromadb")

        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self._embedder_available = True
            logger.info(f"Embedding model loaded: {settings.EMBEDDING_MODEL}")
        except ImportError:
            logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if self._embedder_available and self._embedding_model:
            embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        # Fallback: simple hash-based embeddings for demo
        return [self._simple_embedding(t) for t in texts]

    def _simple_embedding(self, text: str) -> List[float]:
        """Simple fallback embedding for when sentence-transformers isn't available."""
        import hashlib
        vector = []
        for i in range(384):
            h = hashlib.md5(f"{text}{i}".encode()).hexdigest()
            val = int(h[:8], 16) / 0xFFFFFFFF
            vector.append(val)
        return vector

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection."""
        if name not in self._collections:
            if self._chroma_available:
                try:
                    collection = self._chroma_client.get_collection(name)
                except Exception:
                    collection = self._chroma_client.create_collection(name)
                self._collections[name] = collection
            else:
                self._collections[name] = InMemoryCollection(name)
        return self._collections[name]

    def index_policy(self, policy_id: int, policy_text: str, metadata: Dict[str, Any]) -> bool:
        """
        Index an insurance policy document.

        Args:
            policy_id: Unique policy identifier.
            policy_text: Full policy terms text.
            metadata: Additional metadata (policyholder, coverage type, etc.).

        Returns:
            True if indexing succeeded.
        """
        try:
            collection = self._get_or_create_collection(settings.CHROMA_COLLECTION_POLICIES)
            embedding = self._get_embeddings([policy_text])[0]

            collection.add(
                ids=[f"policy_{policy_id}"],
                embeddings=[embedding],
                documents=[policy_text],
                metadatas=[{
                    **metadata,
                    "type": "policy",
                    "policy_id": policy_id,
                }],
            )
            logger.info(f"Indexed policy {policy_id} in vector DB.")
            return True
        except Exception as e:
            logger.error(f"Failed to index policy {policy_id}: {e}")
            return False

    def index_claim(self, claim_id: int, claim_text: str, metadata: Dict[str, Any]) -> bool:
        """
        Index a claim for semantic search.

        Args:
            claim_id: Unique claim identifier.
            claim_text: Claim description and notes text.
            metadata: Additional metadata (category, status, fraud score, etc.).

        Returns:
            True if indexing succeeded.
        """
        try:
            collection = self._get_or_create_collection(settings.CHROMA_COLLECTION_CLAIMS)
            embedding = self._get_embeddings([claim_text])[0]

            collection.add(
                ids=[f"claim_{claim_id}"],
                embeddings=[embedding],
                documents=[claim_text],
                metadatas=[{
                    **metadata,
                    "type": "claim",
                    "claim_id": claim_id,
                }],
            )
            logger.info(f"Indexed claim {claim_id} in vector DB.")
            return True
        except Exception as e:
            logger.error(f"Failed to index claim {claim_id}: {e}")
            return False

    def index_few_shot_example(self, example_id: str, example_text: str, metadata: Dict[str, Any]) -> bool:
        """
        Index a few-shot example for RAG queries.

        Args:
            example_id: Unique identifier for the example.
            example_text: Example query and response text.
            metadata: Additional metadata (category, use case, etc.).

        Returns:
            True if indexing succeeded.
        """
        try:
            collection = self._get_or_create_collection(settings.CHROMA_COLLECTION_FEW_SHOT)
            embedding = self._get_embeddings([example_text])[0]

            collection.add(
                ids=[f"fewshot_{example_id}"],
                embeddings=[embedding],
                documents=[example_text],
                metadatas=[{
                    **metadata,
                    "type": "few_shot",
                    "example_id": example_id,
                }],
            )
            logger.info(f"Indexed few-shot example {example_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to index few-shot example {example_id}: {e}")
            return False

    def search_similar_claims(
        self,
        query_text: str,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for claims similar to the query text.

        Args:
            query_text: Natural language query describing the claim to match.
            top_k: Number of results to return.
            filter_criteria: Optional metadata filters.

        Returns:
            List of matching documents with scores and metadata.
        """
        try:
            collection = self._get_or_create_collection(settings.CHROMA_COLLECTION_CLAIMS)
            query_embedding = self._get_embeddings([query_text])[0]

            where = self._build_filter(filter_criteria) if filter_criteria else None

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 20),
                where=where,
            )

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def search_policies(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search insurance policies by semantic similarity.

        Args:
            query_text: Natural language query about policy terms.
            top_k: Number of results to return.

        Returns:
            List of matching policies with scores.
        """
        try:
            collection = self._get_or_create_collection(settings.CHROMA_COLLECTION_POLICIES)
            query_embedding = self._get_embeddings([query_text])[0]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 20),
            )

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Policy search failed: {e}")
            return []

    def get_few_shot_examples(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve few-shot examples relevant to the query.

        Args:
            query_text: Query to match examples against.
            top_k: Number of examples to retrieve.

        Returns:
            List of few-shot examples with scores.
        """
        try:
            collection = self._get_or_create_collection(settings.CHROMA_COLLECTION_FEW_SHOT)
            query_embedding = self._get_embeddings([query_text])[0]

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, 10),
            )

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Few-shot retrieval failed: {e}")
            return []

    def rag_query(
        self,
        query: str,
        top_k: int = 5,
        collection: str = "historical_claims",
    ) -> Dict[str, Any]:
        """
        Perform a full RAG query: retrieve context and return results.

        Args:
            query: Natural language query.
            top_k: Number of context documents to retrieve.
            collection: Which collection to search (historical_claims, insurance_policies, few_shot_examples).

        Returns:
            Dict with query, retrieved context, and formatted results.
        """
        collection_map = {
            "historical_claims": self.search_similar_claims,
            "insurance_policies": self.search_policies,
            "few_shot_examples": self.get_few_shot_examples,
        }

        search_fn = collection_map.get(collection, self.search_similar_claims)
        results = search_fn(query, top_k=top_k)

        return {
            "query": query,
            "collection": collection,
            "results": results,
            "result_count": len(results),
        }

    def _build_filter(self, criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB filter from criteria dict."""
        if not criteria:
            return None

        filters = {}
        for key, value in criteria.items():
            if isinstance(value, (str, int, float, bool)):
                filters[key] = value
            elif isinstance(value, dict):
                filters[key] = value  # Pass through operators

        return filters if filters else None

    def _format_results(self, results) -> List[Dict[str, Any]]:
        """Format ChromaDB query results into clean dicts."""
        formatted = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted

        for i in range(len(results["ids"][0])):
            item = {
                "id": results["ids"][0][i],
                "score": float(results["distances"][0][i]) if results.get("distances") else 0.0,
                "document": results["documents"][0][i][:500] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            }
            formatted.append(item)

        # Sort by score (lower distance = more similar)
        formatted.sort(key=lambda x: x["score"])
        return formatted

    def index_all_claims_from_db(self) -> int:
        """Index all existing claims from the database into the vector store."""
        from backend.database import SessionLocal, Claim, ClaimNote, Document

        session = SessionLocal()
        try:
            claims = session.query(Claim).all()
            indexed = 0

            for claim in claims:
                notes = session.query(ClaimNote).filter(ClaimNote.claim_id == claim.id).all()
                docs = session.query(Document).filter(Document.claim_id == claim.id).all()

                text_parts = [
                    f"Claim {claim.claim_number}",
                    f"Category: {claim.category}",
                    f"Status: {claim.status}",
                    f"Description: {claim.description or ''}",
                ]
                for note in notes:
                    text_parts.append(f"Note: {note.content}")
                for doc in docs:
                    text_parts.append(f"Document: {doc.ocr_text or ''}")

                claim_text = "\n".join(text_parts)

                self.index_claim(
                    claim_id=claim.id,
                    claim_text=claim_text,
                    metadata={
                        "claim_number": claim.claim_number,
                        "category": claim.category,
                        "status": claim.status,
                        "fraud_score": claim.fraud_score,
                        "policyholder": claim.policyholder_name,
                    },
                )
                indexed += 1

            return indexed

        finally:
            session.close()

    def index_few_shot_defaults(self) -> int:
        """Index default few-shot examples for common claim scenarios."""
        examples = [
            {
                "id": "auto_damage_assessment",
                "text": "Claim: Rear-end collision on highway. Front bumper and radiator damaged. "
                        "Driver reported whiplash. Police report filed. "
                        "Action: Assess vehicle damage, verify police report, check medical records for whiplash treatment.",
                "category": "auto",
                "use_case": "damage_assessment",
            },
            {
                "id": "health_claim_verification",
                "text": "Claim: Emergency room visit for chest pain. Diagnosed with angina. "
                        "Policy covers cardiac events with 10% copay. "
                        "Action: Verify ER visit date matches claim, ensure policy was active, process with cardiac coverage terms.",
                "category": "health",
                "use_case": "verification",
            },
            {
                "id": "property_fraud_check",
                "text": "Claim: Water damage from burst pipe. Policyholder claims extensive kitchen damage. "
                        "Note: Claim filed 45 days after incident. Previous water damage claim 3 months ago. "
                        "Action: Flag for delay in filing, investigate prior claim similarity, schedule adjuster visit.",
                "category": "property",
                "use_case": "fraud_detection",
            },
            {
                "id": "travel_claim_processing",
                "text": "Claim: Trip cancellation due to family emergency. "
                        "Policy covers cancellation for medical emergencies of immediate family. "
                        "Action: Verify family relationship, check medical documentation, confirm cancellation fees.",
                "category": "travel",
                "use_case": "processing",
            },
            {
                "id": "life_claim_documentation",
                "text": "Claim: Accidental death benefit. Policyholder died in car accident. "
                        "Beneficiary is spouse. Policy requires death certificate and police report. "
                        "Action: Verify beneficiary identity, obtain certified death certificate, review policy exclusions.",
                "category": "life",
                "use_case": "documentation",
            },
        ]

        indexed = 0
        for ex in examples:
            try:
                self.index_few_shot_example(
                    example_id=ex["id"],
                    example_text=ex["text"],
                    metadata={"category": ex["category"], "use_case": ex["use_case"]},
                )
                indexed += 1
            except Exception as e:
                logger.warning(f"Failed to index few-shot example {ex['id']}: {e}")

        return indexed


class InMemoryCollection:
    """
    Fallback in-memory collection for when ChromaDB isn't available.
    Implements basic similarity search using the simple embedding function.
    """

    def __init__(self, name: str):
        self.name = name
        self._ids: List[str] = []
        self._embeddings: List[List[float]] = []
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []

    def add(self, ids, embeddings, documents, metadatas):
        for i in range(len(ids)):
            self._ids.append(ids[i])
            self._embeddings.append(embeddings[i])
            self._documents.append(documents[i])
            self._metadatas.append(metadatas[i])

    def query(self, query_embeddings, n_results, where=None):
        import math

        query_vec = query_embeddings[0]
        scored = []

        for i in range(len(self._ids)):
            if where and not self._matches_where(self._metadatas[i], where):
                continue

            emb = self._embeddings[i]
            dist = 1.0 - sum(a * b for a, b in zip(query_vec, emb)) / (
                math.sqrt(sum(a * a for a in query_vec)) * math.sqrt(sum(b * b for b in emb)) + 1e-10
            )
            scored.append((dist, i))

        scored.sort(key=lambda x: x[0])
        top_k = scored[:n_results]

        return {
            "ids": [[self._ids[i] for _, i in top_k]],
            "distances": [[d for d, _ in top_k]],
            "documents": [[self._documents[i] for _, i in top_k]],
            "metadatas": [[self._metadatas[i] for _, i in top_k]],
        }

    def _matches_where(self, meta, where):
        for key, value in where.items():
            if key not in meta or meta[key] != value:
                return False
        return True


# Module-level singleton
rag = RAGPipeline()
