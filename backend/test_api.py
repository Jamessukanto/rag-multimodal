"""
Quick test script for the MCP Client API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_tools():
    """Test /tools endpoint"""
    print("=" * 50)
    print("Testing /tools endpoint...")
    print("=" * 50)
    try:
        response = requests.get(f"{BASE_URL}/tools")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Number of tools: {len(data.get('tools', []))}")
            print(f"Tools: {json.dumps(data, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    print()

def test_query(query: str, library: str = "langchain"):
    """Test /query endpoint with a documentation query"""
    print("=" * 50)
    print(f"Testing /query endpoint...")
    print(f"Query: '{query}'")
    print(f"Library: {library}")
    print("=" * 50)
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"query": query},
            timeout=60  # Longer timeout for LLM + tool calls
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            messages = data.get("messages", [])
            print(f"\nNumber of messages: {len(messages)}")
            
            # Print conversation
            for i, msg in enumerate(messages, 1):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                print(f"\n--- Message {i} ({role}) ---")
                if isinstance(content, str):
                    print(content[:500] + "..." if len(content) > 500 else content)
                else:
                    print(json.dumps(content, indent=2)[:500] + "...")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    print()

if __name__ == "__main__":
    print("Starting API tests for MCP Documentation Client...")
    print()
    
    # Test 1: Get available tools
    test_tools()
    
    # Test 2: Query about LangChain + ChromaDB (should use MCP tools)
    test_query("How to use ChromaDB with LangChain?")
    
    # Test 3: Query about vector stores
    test_query("How do I create a vector store in LangChain?")
    
    # Test 4: Query about OpenAI embeddings
    test_query("How to use OpenAI embeddings in LangChain?")
    
    print("Tests complete!")
    print("\nNote: These queries should trigger MCP tools to search documentation.")
    print("The LLM will use the available tools to find relevant docs.")

