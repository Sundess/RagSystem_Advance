import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PineconeManager:
    """
    Simplified Pinecone manager with persistent embeddings check
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

    def query_documents(self, query, k=3):
        """Query the vectorstore for relevant documents (alias for compatibility)"""
        return self.search_documents(query, k)

    def search_documents(self, query, k=3):
        """Search the vectorstore for relevant documents"""
        if not self.vectorstore:
            return []

        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            if results:
                print(f"📋 Found {len(results)} relevant documents")
                for i, (doc, score) in enumerate(results, 1):
                    print(
                        f"  {i}. Similarity: {1-score:.3f} | Preview: {doc.page_content[:100]}...")

            return [doc for doc, score in results]

        except Exception as e:
            print(f"❌ Error querying documents: {e}")
            return []

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
