import asyncio
import aiohttp
import pydantic


async def chat_completion(query):
    url = "http://localhost:8000/stream/"  # Adjust as needed
    async with aiohttp.ClientSession() as session:
        # Make a POST request and await the response
        print("Sending request to chat API")
        async with session.post(url, json={"message": query}) as response:

            # Stream the response
            async for data_chunk in response.content.iter_chunked(1024):
                text_chunk = data_chunk.decode("utf-8")
                print(text_chunk, end="")

if __name__ == "__main__":
    asyncio.run(chat_completion("Question"))