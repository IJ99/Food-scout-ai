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

## 📌 API Endpoints

### 🔍 Food & Restaurant Search
- **POST `/extract`** → Extract food and city from user input  
- **POST `/search-restaurants`** → Search restaurants by food + city  
- **POST `/full-search`** → Full flow (user check, extraction, geocoding, fallback, history)

### 📜 User History & Recommendations
- **GET `/history/{email}`** → Get past search history  
- **GET `/recommend/{email}`** → Recommend foods based on user history  

### 💬 Chat & Conversation
- **POST `/chat`** → Simple chat interface with memory  
- **POST `/chat-smart`** → Smart chat with food inference + restaurant suggestions  
- **GET `/memory/{session_id}`** → Retrieve chat history for a session  
- **GET `/summary/{session_id}`** → Summarize foods mentioned in a session  

### 📲 Notifications
- **POST `/test-sms`** → Send test SMS via Twilio  
- **POST `/test-email`** → Send test Email via EmailJS  

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
