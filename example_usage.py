"""Example usage of the multi-agent system."""
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def example_text_query():
    """Example: Text-only query."""
    print("=" * 50)
    print("Example 1: Text Query")
    print("=" * 50)
    
    response = requests.post(
        f"{BASE_URL}/query",
        json={
            "query": "What is the capital of France? Explain why it's significant.",
            "use_rag": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Agent Used: {result['agent_used']}")
        print(f"Verified: {result['verified']}")
        print(f"Sources: {result.get('sources', [])}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def example_multimodal_query():
    """Example: Multimodal query with image."""
    print("\n" + "=" * 50)
    print("Example 2: Multimodal Query")
    print("=" * 50)
    
    # Note: Replace with actual image path
    image_path = "path/to/your/image.jpg"
    
    try:
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"query": "What is in this image? Describe it in detail."}
            
            response = requests.post(
                f"{BASE_URL}/query-multimodal",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result['response']}")
                print(f"Agent Used: {result['agent_used']}")
                print(f"Verified: {result['verified']}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        print("Please provide a valid image path.")


def example_add_documents():
    """Example: Add documents to RAG."""
    print("\n" + "=" * 50)
    print("Example 3: Add Documents to RAG")
    print("=" * 50)
    
    documents = [
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "France is a country in Western Europe with a rich cultural history."
    ]
    
    response = requests.post(
        f"{BASE_URL}/rag/add-documents",
        json={"documents": documents}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['message']}")
        print(f"Total Documents: {result['total_documents']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def example_health_check():
    """Example: Health check."""
    print("\n" + "=" * 50)
    print("Example 4: Health Check")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")
        print(f"Components: {json.dumps(result['components'], indent=2)}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    print("Multi-Agent System - Example Usage\n")
    
    # Check if server is running
    try:
        example_health_check()
        print("\n")
        
        # Run examples
        example_add_documents()
        example_text_query()
        # example_multimodal_query()  # Uncomment when you have an image
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Please make sure the server is running:")
        print("  python main.py")
    except Exception as e:
        print(f"Error: {e}")

