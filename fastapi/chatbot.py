from typing import List, Optional
import uuid
import os

import chromadb
from openai import OpenAI

import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings

class ChatBot:
    """
    A ChatBot that can vectorize a set of files (by chunking them)
    and then store/retrieve embeddings in a Chroma DB collection.
    The retrieved chunks are used as context for GPT-based chat responses.
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
        :param api_key:      Your OpenAI API key, or None to use env var OPENAI_API_KEY.
        :param chat_model:   The chat model for OpenAI ChatCompletion (e.g. 'gpt-3.5-turbo').
        :param embedding_model: The embedding model for OpenAI (e.g. 'text-embedding-ada-002').
        :param persist_directory: Directory for local Chroma DB persistence.
        :param collection_name:   The name of the Chroma collection to store embeddings.
        """
        # Prefer explicit api_key or fallback to environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.chat_model = chat_model or os.getenv("MODEL", "gpt-4o-mini")
        self.embedding_model = (
            embedding_model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        )

        # Configure OpenAI

        # Create a Chroma client (DuckDB+Parquet by default).
        self.client = chromadb.Client(
            Settings(persist_directory=persist_directory)
        )

        # Create an OpenAI embedding function that Chroma can call automatically
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key,
            model_name=self.embedding_model
        )

        # Get or create a persistent collection in Chroma
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def _read_file(self, file_path: str) -> str:
        """
        Read the entire file content into a string.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Chunk the text into overlapping segments. Overlaps help the model
        preserve context between chunks.
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
        Reads the given files, splits them into chunks, and adds them to the Chroma DB collection.
        Chroma will automatically use the provided embedding function to vectorize the chunks.

        :param file_paths: List of file paths to ingest.
        :param chunk_size: Number of characters in each chunk.
        :param overlap: Overlap between consecutive chunks.
        """
        for path in file_paths:
            try:
                text = self._read_file(path)
                chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)

                # Collect the chunks to add
                ids = []
                docs = []
                metas = []

                for chunk in chunks:
                    doc_id = str(uuid.uuid4())  # unique ID for each chunk
                    ids.append(doc_id)
                    docs.append(chunk)
                    metas.append({"source": path})

                # Add them to the collection.
                # Chroma automatically calls our OpenAIEmbeddingFunction behind the scenes.
                if ids:  # Only add if non-empty
                    self.collection.add(
                        documents=docs,
                        metadatas=metas,
                        ids=ids
                    )

            except Exception as e:
                print(f"Error processing file {path}: {e}")

    def _get_relevant_chunks(self, query: str, top_k: int = 3) -> List[str]:
        """
        Given a user query, retrieve the top_k most relevant chunks from Chroma.

        Since we configured `embedding_function` on the collection,
        we can pass `query_texts` directly instead of embedding ourselves.
        """
        try:
            # Use query_texts instead of query_embeddings
            results = self.collection.query(
                query_texts=[query],  # We only have one query here
                n_results=top_k
            )
            # The 'documents' field is a list of lists (one list per query).
            # For a single query, results['documents'][0] is the chunk list.
            return results["documents"][0]
        except Exception as e:
            print(f"Error retrieving relevant chunks: {e}")
            return []

    def ask(self, user_input: str, top_k: int = 3, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate a short and precise answer to `user_input` using the top_k relevant chunks
        from the Chroma DB as context.

        :param user_input: The user's question or prompt.
        :param top_k: Number of chunks to retrieve for context.
        :param max_tokens: Max tokens for OpenAI to generate.
        :param temperature: Sampling temperature for creativity.
        :return: The model's answer.
        """
        # 1) Retrieve context from Chroma
        relevant_chunks = self._get_relevant_chunks(query=user_input, top_k=top_k)

        # 2) Combine the relevant chunks into a single context string
        context_text = "\n".join(relevant_chunks)

        # 3) Build a system prompt
        system_prompt = (
            "You are a helpful assistant that provides short and precise answers. "
            "Below is some relevant context:\n"
            f"{context_text}\n\n"
            "Use this context to answer the user's question accurately and concisely."
            "Use only plaintext, no formatting and no markup"
        )

        try:
            # 4) Call the OpenAI ChatCompletion endpoint

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            max_tokens=max_tokens,
            temperature=temperature)
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error generating chat completion: {e}")
            return "Sorry, I couldn't process your request."
