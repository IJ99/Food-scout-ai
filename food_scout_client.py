import requests
import json

class FoodScoutClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def search_food(self, user_input):
        """Search for food using natural language"""
        try:
            response = requests.post(
                f"{self.base_url}/full-search",
                json={"user_input": user_input},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def extract_only(self, user_input):
        """Just extract food and city"""
        try:
            response = requests.post(
                f"{self.base_url}/extract",
                json={"user_input": user_input}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# Usage example
if __name__ == "__main__":
    # Use local server
    client = FoodScoutClient("http://localhost:8000")
    
    # Or use your Render deployment
    # client = FoodScoutClient("https://your-app-name.onrender.com")
    
    print("🍽️ Food Scout AI Client")
    
    while True:
        user_input = input("\nWhat are you craving and where? (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        print("🤖 Searching...")
        result = client.search_food(user_input)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"🍝 Food: {result['extracted']['food']}")
            print(f"📍 City: {result['extracted']['city']}")
            print(f"🏪 Found {result['count']} restaurants:")
            
            for restaurant in result['restaurants'][:5]:  # Show first 5
                print(f"  • {restaurant['name']} - {restaurant['address']}")