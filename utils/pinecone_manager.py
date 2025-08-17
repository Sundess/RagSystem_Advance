import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np


class PineconeManager:
    """
    Enhanced Pinecone manager with document reranking capabilities
    """

    def __init__(self):
        self.pc = None
        self.embeddings = None
        self.index_name = "my-embeddings-index"
        self.vectorstore = None

    def initialize(self):
        """Initialize Pinecone and Google embeddings"""
        print("🔧 Initializing components...")

        # Initialize Google embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        return self.embeddings, self.pc

    def setup_vectorstore(self):
        """Setup vectorstore with existing embeddings check"""
        # Always ensure components are initialized
        if not self.embeddings or not self.pc:
            print("🔄 Initializing components...")
            self.initialize()

        # Check if index exists
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"📝 Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            self._wait_for_index_ready()
        else:
            print(f"✅ Index '{self.index_name}' already exists")

        # Check for existing embeddings
        return self._check_existing_embeddings()

    def _wait_for_index_ready(self):
        """Wait for index to be ready"""
        print("⏳ Waiting for index to be ready...")
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(2)
        print("✅ Index is ready!")

    def _check_existing_embeddings(self):
        """Check if embeddings already exist in the index"""
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            vector_count = stats.total_vector_count

            if vector_count > 0:
                print(f"✅ Found {vector_count} existing embeddings in index")
                # Create vectorstore connection to existing embeddings
                self.vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=self.embeddings
                )
                return self.vectorstore, True  # True = embeddings exist
            else:
                print("📄 Index is empty, no existing embeddings found")
                return None, False  # False = no embeddings

        except Exception as e:
            print(f"❌ Error checking embeddings: {e}")
            return None, False

    def add_documents(self, text_content):
        """Add new documents to vectorstore"""
        print("📄 Processing and adding documents...")

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = splitter.create_documents([text_content])
        print(f"📊 Created {len(docs)} document chunks")

        # Add to vectorstore
        if self.vectorstore:
            # Add to existing vectorstore
            self.vectorstore.add_documents(docs)
        else:
            # Create new vectorstore
            self.vectorstore = PineconeVectorStore.from_documents(
                documents=docs,
                embedding=self.embeddings,
                index_name=self.index_name
            )

        # Wait for indexing
        self._wait_for_indexing(len(docs))
        print(f"✅ Successfully added {len(docs)} documents")

        return self.vectorstore

    def _wait_for_indexing(self, expected_new_docs):
        """Wait for new documents to be indexed"""
        print("⏳ Waiting for documents to be indexed...")
        index = self.pc.Index(self.index_name)

        start_count = index.describe_index_stats().total_vector_count
        target_count = start_count + expected_new_docs

        while True:
            current_count = index.describe_index_stats().total_vector_count
            print(f"  Indexed: {current_count}/{target_count}")

            if current_count >= target_count:
                print("✅ All documents indexed!")
                break
            time.sleep(2)

    def _calculate_bm25_score(self, query_terms, doc_content):
        """
        Calculate BM25 score for a document given query terms
        Simple implementation for reranking
        """
        # Parameters for BM25
        k1, b = 1.5, 0.75

        # Tokenize document (simple word splitting)
        doc_terms = doc_content.lower().split()
        doc_length = len(doc_terms)

        # Estimate average document length (simplified)
        avg_doc_length = 200  # Rough estimate for chunks

        score = 0.0
        for term in query_terms:
            term_lower = term.lower()
            # Count term frequency in document
            tf = doc_terms.count(term_lower)
            if tf > 0:
                # Simple BM25 scoring
                idf_component = 1.0  # Simplified, normally would calculate across corpus
                tf_component = (tf * (k1 + 1)) / (tf + k1 *
                                                  (1 - b + b * (doc_length / avg_doc_length)))
                score += idf_component * tf_component

        return score

    def _calculate_semantic_score(self, query, doc_content):
        """
        Calculate semantic similarity score using keyword matching and content analysis
        """
        query_lower = query.lower()
        doc_lower = doc_content.lower()

        # Exact phrase matching (higher weight)
        exact_matches = 0
        query_words = query_lower.split()
        for i in range(len(query_words) - 1):
            phrase = " ".join(query_words[i:i+2])
            if phrase in doc_lower:
                exact_matches += 2

        # Individual word matching
        word_matches = sum(1 for word in query_words if word in doc_lower)

        # Length normalization
        doc_length = len(doc_content.split())
        # Prefer documents with reasonable length
        length_factor = min(1.0, doc_length / 100)

        # Combine scores
        semantic_score = (exact_matches * 2 + word_matches) * length_factor

        return semantic_score

    def _rerank_documents(self, query, documents_with_scores):
        """
        Rerank documents using hybrid scoring (similarity + BM25 + semantic analysis)
        Returns top 5 documents
        """
        print("🔄 Reranking documents using hybrid scoring...")

        # Prepare query terms for BM25
        query_terms = query.lower().split()

        reranked_docs = []

        for doc, similarity_score in documents_with_scores:
            # Original similarity score (cosine similarity from vector search)
            # Convert distance to similarity (Pinecone returns distance, lower is better)
            # Assuming similarity_score is actually distance
            vector_sim_score = 1 - similarity_score

            # BM25 score
            bm25_score = self._calculate_bm25_score(
                query_terms, doc.page_content)

            # Semantic score
            semantic_score = self._calculate_semantic_score(
                query, doc.page_content)

            # Hybrid score combination with weights
            # You can adjust these weights based on your needs
            hybrid_score = (
                0.4 * vector_sim_score +    # Vector similarity weight
                0.3 * bm25_score +          # BM25 weight
                0.3 * semantic_score        # Semantic analysis weight
            )

            reranked_docs.append((doc, hybrid_score, {
                'vector_score': vector_sim_score,
                'bm25_score': bm25_score,
                'semantic_score': semantic_score,
                'hybrid_score': hybrid_score
            }))

        # Sort by hybrid score (descending)
        reranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Log reranking results
        print("📊 Reranking Results:")
        for i, (doc, final_score, scores) in enumerate(reranked_docs[:5], 1):
            print(f"  {i}. Final Score: {final_score:.3f}")
            print(
                f"     Vector: {scores['vector_score']:.3f} | BM25: {scores['bm25_score']:.3f} | Semantic: {scores['semantic_score']:.3f}")
            print(f"     Preview: {doc.page_content[:100]}...")
            print()

        # Return top 5 documents
        return [doc for doc, score, detailed_scores in reranked_docs[:5]]

    def query_documents(self, query, k=5):
        """Query the vectorstore for relevant documents with reranking (alias for compatibility)"""
        return self.search_documents(query, k)

    def search_documents(self, query, k=5):
        """
        Search the vectorstore for relevant documents with enhanced reranking
        First retrieves k=7 documents, then reranks and returns top 5
        """
        if not self.vectorstore:
            return []

        try:
            # Step 1: Retrieve more documents than needed (k=7)
            print(f"🔍 Retrieving 7 documents for reranking...")
            results = self.vectorstore.similarity_search_with_score(query, k=7)

            if not results:
                print("❌ No documents found")
                return []

            print(f"📋 Found {len(results)} documents for reranking")

            # Step 2: Rerank documents using hybrid scoring
            reranked_docs = self._rerank_documents(query, results)

            # Step 3: Return top 5 reranked documents
            final_count = min(k, len(reranked_docs))
            final_docs = reranked_docs[:final_count]

            print(f"✅ Returning top {final_count} reranked documents")

            return final_docs

        except Exception as e:
            print(f"❌ Error querying documents: {e}")
            return []

    def similarity_search(self, query, k=5):
        """
        Alternative method name for compatibility
        """
        return self.search_documents(query, k)

    def clear_index(self):
        """Clear all vectors from the index"""
        try:
            if self.pc and self.index_name:
                index = self.pc.Index(self.index_name)
                index.delete(delete_all=True)
                self.vectorstore = None
                print("✅ Index cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
            return False

    def get_stats(self):
        """Get index statistics - FIXED VERSION"""
        # Default stats in case of any error
        default_stats = {'total_vectors': 0,
                         'dimension': 768, 'index_fullness': 0.0}

        try:
            # Ensure components are initialized before checking
            if not self.pc or not self.index_name:
                print("⚠️ Pinecone components not initialized, initializing now...")
                self.initialize()

            # Double check after initialization
            if not self.pc or not self.index_name:
                print("❌ Failed to initialize Pinecone components")
                return default_stats

            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()

            result = {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness
            }
            print(f"✅ Got stats successfully: {result}")
            return result

        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return default_stats
