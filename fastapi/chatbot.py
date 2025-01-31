from typing import List, Optional
import uuid
import os

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma

class ChatBot:
    """
    A ChatBot that ingests text files (with chunking) into a Chroma DB
    (managed by LangChain), and can then retrieve the most relevant chunks
    to answer a user’s question using an LLM chain.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            chat_model: Optional[str] = None,
            embedding_model: Optional[str] = None,
            persist_directory: str = "chromadb_storage",
            collection_name: str = "chatbot_embeddings"
    ):
        """
        :param api_key:           Your OpenAI API key, or None to use env var OPENAI_API_KEY.
        :param chat_model:        The chat model (e.g. 'gpt-4o-mini', etc.).
        :param embedding_model:   The embedding model for OpenAI (e.g. 'text-embedding-ada-002').
        :param persist_directory: Directory for local Chroma DB persistence.
        :param collection_name:   The name of the Chroma collection to store embeddings.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.chat_model = chat_model or os.getenv("MODEL", "gpt-4o-mini")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.embedding_function = OpenAIEmbeddings(
            api_key=self.api_key,
            model=self.embedding_model
        )

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    @staticmethod
    def _read_file(file_path: str) -> str:
        """Simple text reader."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Chunk the text into overlapping segments to help with context continuity.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk.strip())
            start += (chunk_size - overlap)

        return chunks

    def vectorize_files(self, file_paths: List[str], chunk_size: int = 1000, overlap: int = 100) -> None:
        """
        Reads the files, splits into chunks, and adds them to the LangChain Chroma store.
        """
        all_texts = []
        all_metadatas = []

        for path in file_paths:
            try:
                text = self._read_file(path)
                chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)

                # Each chunk is stored as a separate "document" in the vector store
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadatas.append({"source": path, "chunk_id": str(uuid.uuid4())})

            except Exception as e:
                print(f"Error processing file {path}: {e}")

        # Add all chunks at once to the vector store
        if all_texts:
            self.vectorstore.add_texts(texts=all_texts, metadatas=all_metadatas)

    def ask(self, user_input: str, top_k: int = 5, max_tokens: int = 200, temperature: float = 0.7) -> dict:
        """
        Generate a concise answer using the top_k retrieved chunks as context.

        :param user_input:  The user's question.
        :param top_k:       Number of chunks to retrieve from the store.
        :param max_tokens:  Max tokens in the final LLM answer.
        :param temperature: Sampling temperature for creativity.
        :return:            The chatbot answer as a string.
        """

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                model=self.chat_model,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens
            ),
            chain_type="stuff",  # You can pick "stuff", "map_reduce", "refine", etc.
            retriever=retriever,
            return_source_documents=False  # Set to True if you also want the source docs
        )

        return qa_chain.invoke(user_input)
