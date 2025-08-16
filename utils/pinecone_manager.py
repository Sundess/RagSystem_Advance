import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


class PineconeManager:
    """
    Manages Pinecone vector database operations
    """

    def __init__(self):
        self.pc = None
        self.embeddings = None
        self.index_name = "my-embeddings-index"

    def setup_pinecone_with_google_embeddings(self):
        """
        Set up Pinecone vector database with Google embeddings
        """
        print("🔧 Initializing Google embeddings...")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        print("🔧 Initializing Pinecone...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        # Check if index exists, create if not
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"📝 Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=768,  # Google's text-embedding-004 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print("⏳ Waiting for index to be ready...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                print("  Index still initializing...")
                time.sleep(2)
            print("✅ Index is ready!")
        else:
            print(f"✅ Using existing index: {self.index_name}")

        return self.embeddings, self.index_name, self.pc

    def process_and_store_documents(self, text_content, embeddings, index_name, pc):
        """
        Process text content and store in Pinecone
        """
        print("📄 Processing documents...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size for better context
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        docs = splitter.create_documents([text_content])
        print(f"📊 Created {len(docs)} document chunks")

        print("🚀 Storing documents in Pinecone...")
        vectorstore = PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name
        )

        print("⏳ Waiting for vectors to be indexed...")
        index = pc.Index(index_name)
        expected_count = len(docs)

        # Wait for all vectors to be indexed
        while True:
            stats = index.describe_index_stats()
            current_count = stats.total_vector_count
            print(f"  Vectors indexed: {current_count}/{expected_count}")

            if current_count >= expected_count:
                print("✅ All vectors have been indexed!")
                break

            time.sleep(2)

        print(
            f"✅ Successfully stored {len(docs)} documents in Pinecone index: {index_name}")
        return vectorstore

    def query_vectorstore(self, vectorstore, query, k=3):
        """
        Query the vector store for relevant documents
        """
        try:
            results_with_scores = vectorstore.similarity_search_with_score(
                query, k=k)

            if not results_with_scores:
                print("❌ No documents found")
                return []

            print(f"📋 Found {len(results_with_scores)} relevant documents")
            for i, (doc, score) in enumerate(results_with_scores, 1):
                print(
                    f"  {i}. Similarity: {1-score:.3f} | Preview: {doc.page_content[:100]}...")

            return [doc for doc, score in results_with_scores]

        except Exception as e:
            print("Error querying vector database", e)

    def clear_index(self):
        """
        Clear all vectors from the Pinecone index
        """
        try:
            if self.pc and self.index_name:
                index = self.pc.Index(self.index_name)
                # Delete all vectors in the index
                index.delete(delete_all=True)
                print("✅ Pinecone index cleared successfully")
            return True
        except Exception as e:
            print(f"❌ Error clearing Pinecone index: {e}")
            return False
