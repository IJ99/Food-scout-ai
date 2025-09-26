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

##🧠 How It Works
🧑‍🍳 User Journey in Food Scout AI

Starting a search

The user types something natural like:
“I feel like eating suya in Abuja”
or
“Where can I get amala in Ibadan?”

Getting results

Food Scout AI understands the request.

It figures out the dish and the city.

It then shows a list of nearby restaurants that serve that dish, along with their names and addresses.

The user also sees how many results were found.

If no restaurant is found, the system suggests something very similar (e.g., another Nigerian dish) and explains why.

History & memory

Every search the user makes is saved to their personal history.

They can later check “my past searches” to see all the foods and cities they asked about.

Smart recommendations

Over time, as the user searches for more foods, the system learns their taste.

The user can ask for recommendations, and Food Scout AI will suggest new dishes they might enjoy, based on their history (not just random foods).

Chat option

Instead of only structured searches, the user can simply “chat” with Food Scout AI.

Example: “I’m craving something spicy” → the system suggests suya or pepper soup and may even show nearby places.

The chat remembers the conversation, so users feel like they’re talking to a personal food assistant.

Notifications

(Optional) The system can also send test SMS or email notifications, like:
“A customer is interested in ordering Amala — please expect a call soon!”

✅ In summary

From the user’s eyes, Food Scout AI feels like:

Ask naturally → “I want amala in Ibadan.”

Get results → Restaurants nearby serving it.

If no luck → Suggestions for similar dishes with an explanation.

Build history → See all your past searches in one place.

Get recommendations → Smart new dishes suggested based on your tastes.

Chat freely → A friendly assistant that helps you decide what to eat.

🔎 POST /extract – Extract Food & City

What it does:
You type something natural like “I want amala in Ibadan.”
The system figures out:

Food = amala

City = Ibadan

When to use: If you just want the system to understand your request without searching restaurants yet.

🍴 POST /search-restaurants – Find Restaurants

What it does:
You give the food name and the city, and it shows restaurants nearby.

Example:
Send { "food": "pizza", "city": "Lagos" } → You get a list of pizza places in Lagos.

🚀 POST /full-search – One-Step Everything

What it does:
This is the main shortcut.

You type a natural request like “I feel like eating suya in Abuja.”

The system:

Figures out food + city.

Finds restaurants.

Saves your search in history.

If nothing is found, it suggests similar dishes instead.

Best for: Most users, since it’s a single request from start to finish.

📜 History & Recommendations
🕒 GET /history/{email} – Past Searches

What it does:
Shows all the foods and cities you’ve searched before, tied to your email.

Why it matters: Helps you remember what you looked for in the past.

🍲 GET /recommend/{email} – Personalized Suggestions

What it does:
Looks at your search history and suggests new dishes you might enjoy.

Example:
If you searched for suya, jollof rice, and okra soup, it might recommend egusi soup or yam porridge.

Why it’s special: It’s not random — it’s based on your personal tastes.

💬 Chat & Conversation
💭 POST /chat – Regular Chat

What it does:
Lets you have a conversation with the AI about food.
The chat remembers your session, so it feels like talking to a personal assistant.

🧠 POST /chat-smart – Smarter Chat

What it does:
More intelligent than /chat.

If you say something vague like “I’m hungry, I want something spicy”,
it will guess a food (e.g., pepper soup) and even show nearby places.

📚 GET /memory/{session_id} – See Chat Memory

What it does:
Shows the whole conversation history for that chat session.

📝 GET /summary/{session_id} – Food Summary

What it does:
Quickly lists all the foods mentioned in a session.
Example: “suya, amala, shawarma”

📢 Notifications
📱 POST /test-sms – SMS Test

What it does:
Sends a sample text message (via Twilio) to confirm the SMS feature works.

📧 POST /test-email – Email Test

What it does:
Sends a sample email (via EmailJS) to confirm the email feature works.

##📦 Project Structure
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

##🙏 Acknowledgements
Groq + LLaMA

Geoapify

FastAPI

Twilio

EmailJS
