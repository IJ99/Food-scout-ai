# 🍽️ Food Scout AI

Food Scout AI is a FastAPI-based backend service that helps users find nearby restaurants serving their desired meals using natural language queries. It uses LLaMA (via Groq API) for language understanding, Geoapify for geolocation and restaurant search, and SQLite for user and search history storage. It also includes Twilio/EmailJS integrations to notify restaurants.

---

## 🚀 Features
- 🤖 AI-powered food and city extraction from user input  
- 📍 Geo-based restaurant recommendations using Geoapify  
- 🔄 Fallback logic using semantic search and LLaMA  
- 🗂️ Persistent user and history tracking via SQLite  
- 📲 Restaurant notifications via SMS (Twilio) or Email (EmailJS)  
- 💬 Chat interface with memory using conversation sessions  

---

## 🛠️ Tech Stack
- **FastAPI** for REST API  
- **Groq + LLaMA3** for natural language understanding  
- **Geoapify** for restaurant search  
- **Twilio / EmailJS** for notifications  
- **SQLite** for local persistence  
- **Docker** for deployment  

---

## 📖 API Documentation
Interactive Swagger docs are available here:  
👉 [https://food-scout-ai.onrender.com/docs](https://food-scout-ai.onrender.com/docs)

---

## 🔑 Example API Endpoints

### `POST /extract`
Extract food and city from user input  
Request:
```json

{ "user_input": "I want amala in Ibadan", "name": "Joel", "email": "joel@email.com" }
Response:

json
Copy code
{ "food": "amala", "city": "Ibadan", "success": true }
POST /search-restaurants
Search restaurants by food & city
Request:

json
Copy code
{ "food": "pizza", "city": "Lagos", "radius": 5000 }
POST /full-search
Does everything: user check, extraction, geocoding, fallback, history

🧠 How It Works
User sends natural query like "I want amala in Ibadan"

API extracts food + city via Groq

Coordinates are fetched from Geoapify

Restaurants are searched and filtered

If no match → fallback via semantic similarity or LLaMA suggestion

Result is saved + returned with nearby matches

📦 Project Structure
graphql
Copy code
.
├── main.py               # Core FastAPI app
├── db_utils.py           # DB helpers (users, history)
├── start.sh              # Startup script
├── Dockerfile            # Docker build
├── render.yaml           # Render deploy settings
├── food_scout_client.py  # Optional CLI test client
├── requirements.txt      # Dependencies
├── runtime.txt           # Python version
├── food_scout.db         # SQLite DB (auto-created)

🙏 Acknowledgements
Groq + LLaMA

Geoapify

FastAPI

Twilio

EmailJS
