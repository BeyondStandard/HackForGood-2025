from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma

import asyncio
import logging
import typing
import uuid
import os

logger = logging.getLogger("uvicorn.error")


class TokenCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        super().__init__()
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

    def request_token(self) -> typing.Optional[str]:
        if self.tokens:
            return self.tokens.pop(0)
        return None


class ChatBot:
    def __init__(
        self,
        api_key: typing.Optional[str] = None,
        chat_model: typing.Optional[str] = None,
        embedding_model: typing.Optional[str] = None,
        persist_directory: str = "chromadb_storage",
        collection_name: str = "chatbot_embeddings",
    ):
        """
        :param api_key:           Your OpenAI API key, or None to use env var OPENAI_API_KEY.
        :param chat_model:        The LLM model name. E.g. "gpt-3.5-turbo", "gpt-4o-mini", etc.
        :param embedding_model:   The embedding model (e.g. "text-embedding-ada-002").
        :param persist_directory: Directory for local Chroma DB persistence.
        :param collection_name:   The name of the Chroma collection to store embeddings.
        """

        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.chat_model = chat_model or os.getenv("MODEL", "gpt-4o-mini")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Build prompt template
        self.prompt = ChatPromptTemplate([
            (
                "system",
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise. "
                "The answer will be used by TTS so avoid complex formats like URLs, or times like '19:00'; "
                "use plain expressions (e.g. '7 PM').\n"
                "Context: {context}\n\n"
                "Even if you know the answer but it's not in the context, say you don't think you can help with that."
            ),
            ("human", "{input}")
        ])

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
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def vectorize_files(self, file_paths: list[str], chunk_size: int = 1000, overlap: int = 100) -> None:
        all_texts = []
        all_metadatas = []

        for path in file_paths:
            try:
                text = self._read_file(path)
                chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for chunk in chunks:
                    all_texts.append(chunk)
                    all_metadatas.append({"source": path, "chunk_id": str(uuid.uuid4())})
            except Exception as e:
                logger.error(f"Error processing file {path}: {e}")

        if all_texts:
            self.vectorstore.add_texts(texts=all_texts, metadatas=all_metadatas)

    def _create_chain_and_callback(self) -> tuple:
        callback = TokenCallbackHandler()
        llm = ChatOpenAI(
            model=self.chat_model,
            streaming=True,
            api_key=self.api_key,
            callbacks=[callback],
            temperature=0.7,
            max_tokens=500,
        )

        stuff_documents_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=self.prompt
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        rag_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=stuff_documents_chain,
        )

        return rag_chain, callback

    async def ask(self, user_input: str) -> str:
        chain, _ = self._create_chain_and_callback()

        def sync_invoke(u_i):
            return chain.invoke({"input": u_i})

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, sync_invoke, user_input)
        return result.get("answer", "")

    async def stream_ask(self, user_input: str) -> typing.AsyncGenerator[str, None]:
        chain, callback = self._create_chain_and_callback()

        complete = False

        def sync_invoke(u_i):
            nonlocal complete
            chain.invoke({"input": u_i})
            complete = True

        loop = asyncio.get_running_loop()
        _ = loop.run_in_executor(None, sync_invoke, user_input)

        while not complete:
            if token := callback.request_token():
                yield token
            else:
                await asyncio.sleep(0.05)

        while True:
            token = callback.request_token()
            if token:
                yield token
            else:
                break
