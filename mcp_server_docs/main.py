from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv 
import httpx
import json
import os
from bs4 import BeautifulSoup

load_dotenv()

mcp = FastMCP('documentation')

USER_AGENT = "docs-app/1.0"
SERPER_URL = "https://google.serper.dev/search"
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# TAVILY_URL = "https://api.tavily.com/search"

docs_url = {
    "langchain": "python.langchain.com/docs", 
    "llama-index": "docs.llamaindex.ai/en/stable",
    "openai": "platform.openai.com/docs",
}


async def search_web(query: str) -> dict | None:
    payload = json.dumps({
        "q": query,
        "num": 2
    })
    headers = {
        "X-API-KEY": os.getenv("SERPER_API_KEY"),
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                SERPER_URL, headers=headers, data=payload, timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error: {e}"}
        except Exception as e:
            return {"error": f"An error occurred: {e}"}


async def fetch_url(url: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            response = await client.get(
                url,
                timeout=30.0,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text()
            return text
        except httpx.HTTPStatusError as e:
            return f"HTTP error {e.response.status_code}: {e}"
        except httpx.TimeoutException:
            return "Timeout error"
        except Exception as e:
            return f"Error fetching URL: {e}"


@mcp.tool()
async def get_docs(query: str, library: str = "langchain"):
    """
    Search the docs for a given query and library.
    Supports langchain, openai, and llama-index.
    
    Args:
        query: The query to search for (e.g. "Chroma DB")
        library: The library to search in (e.g. "langchain")
    
    Returns:
        List of dictionaries containing source URLs and extracted text
    """
    if library not in docs_url:
        raise ValueError(f"Unknown library. Choose from: {list(docs_url.keys())}")
    
    site_filter = docs_url[library]
    search_query = f"site:{site_filter} {query}"
    results = await search_web(search_query)
    
    # Check for errors from Serper
    if "error" in results:
        return f"Search error: {results['error']}"
    
    # Check if organic results exist
    if "organic" not in results or not results["organic"]:
        return "No results found"
    
    # Fetch content from top 3 results 
    text_parts = []
    for result in results["organic"][:3]:
        if "link" in result:
            content = await fetch_url(result["link"])
            text_parts.append(f"URL: {result['link']}\n{content[:4000]}...")
    output = "\n\n---\n\n".join(text_parts)
    return output if output else "No results found"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
