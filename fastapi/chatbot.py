from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma

import asyncio
import typing
import uuid
import os


class TokenCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def request_token(self):
        if self.tokens:
            return self.tokens.pop(0)


class ChatBot:
    """
    A ChatBot that ingests text files (with chunking) into a Chroma DB
    (managed by LangChain), and can then retrieve the most relevant chunks
    to answer a user’s question using an LLM chain.
    """

    def __init__(
            self,
            api_key: typing.Optional[str] = None,
            chat_model: typing.Optional[str] = None,
            embedding_model: typing.Optional[str] = None,
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
        self.complete = None
        self.callback = TokenCallbackHandler()
        self.prompt = ChatPromptTemplate([
            (
                "system",
                "You are an assistant for question-answering tasks."
                "Use the following pieces of retrieved context to answer the question."
                "If you don't know the answer, just say that you don't know."
                "Use three sentences maximum and keep the answer concise.\n"
                "Context: {context}\n\n"
                "Even if you know the answer to the question but it's not mentioned"
                "in the context, you should say you don't think you can help with that."
            ),
            ("human", "{input}")
        ])

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

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

        stuff_documents_chain = create_stuff_documents_chain(
            llm=ChatOpenAI(
                model=self.chat_model,
                streaming=True,
                api_key=self.api_key,
                callbacks=[self.callback],
                temperature=0.7,
                max_tokens=1000
            ),
            prompt=self.prompt
        )

        self.rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=stuff_documents_chain,
        )

    @staticmethod
    def _read_file(file_path: str) -> str:
        """Simple text reader."""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
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

    def vectorize_files(self, file_paths: list[str], chunk_size: int = 1000, overlap: int = 100) -> None:
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

    def _ask(self, message: str) -> dict[str, typing.Any]:
        return self.rag_chain.invoke({"input": message})

    async def _response(self):
        while not self.complete:
            if token := self.callback.request_token():
                yield token

            else:
                await asyncio.sleep(0)

    async def ask(self, user_input: str) -> dict[str, typing.Any]:
        self.complete = False
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._ask, user_input)

        self.complete = True
        return result

    async def response(self):
        async for item in self._response():
            yield item


