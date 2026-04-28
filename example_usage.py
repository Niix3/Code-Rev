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
        example_text_query()
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Please make sure the server is running:")
        print("  python main.py")
    except Exception as e:
        print(f"Error: {e}")

