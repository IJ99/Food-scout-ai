 Food Scout AI:

Food Scout AI is a FastAPI-based backend service that helps users find nearby restaurants serving their desired meals using natural language queries. It uses LLaMA (via Groq API) for language understanding, Geoapify for geolocation and restaurant search, and SQLite for user and search history storage. It also includes Twilio/EmailJS integrations to notify restaurants.


 Features:

*  AI-powered food and city extraction from user input
*  Geo-based restaurant recommendations using Geoapify
*  Fallback logic using semantic search and LLaMA
*  Persistent user and history tracking via SQLite
*  Restaurant notifications via SMS (Twilio) or Email (EmailJS)
*  Chat interface with memory using conversation sessions

 Tech Stack:

* FastAPI for REST API
* Groq + LLaMA3 for natural language understanding
* Geoapify for restaurant search
* Twilio / EmailJS for notifications
* SQLite for local persistence
* Docker for deployment



 API Endpoints:

 # `GET /`

* Health check root
* Response:

```json
{ "message": "Food Scout AI API is running!", "status": "healthy" }
```

# `POST /extract`

* Extract food and city from user input
* Request:

```json
{ "user_input": "I want amala in Ibadan", "name": "Joel", "email": "joel@email.com" }
```

* Response:

```json
{ "food": "amala", "city": "Ibadan", "success": true }
```

# `POST /search-restaurants`

* **Search restaurants by food & city**
* Request:

```json
{ "food": "pizza", "city": "Lagos", "radius": 5000 }
```

* Response:

```json
{ "restaurants": [...], "count": 10 }
```

# `POST /full-search`

* **Does everything**: user check, extraction, geocoding, fallback, history
* Request:

```json
{ "user_input": "I feel like eating suya in Abuja", "name": "Joel", "email": "joel@email.com" }
```

* Response:

```json
{
  "extracted": { "food": "suya", "city": "Abuja" },
  "restaurants": [...],
  "count": 5,
  "message": "Here are results for 'suya' in Abuja"
}
```

# `GET /history/{email}`

* **Get past search history**
* Response:

```json
{
  "user": { "name": "Joel", "email": "joel@email.com" },
  "history": [...],
  "count": 4
}
```

# `GET /recommend/{email}`

* Recommend foods based on user history

# `POST /chat`

* LLM chat interface with memory
* Request:

```json
{ "session_id": "abc123", "message": "I want something spicy" }
```

* Response:

```json
{ "session_id": "abc123", "response": "How about suya?", "inferred_food": null }
```

# `POST /chat-smart`

* Smart chat with food inference and restaurant suggestions

# `GET /memory/{session_id}`

* Retrieve chat history for a session

# `GET /summary/{session_id}`

* Summarize foods mentioned in a session

# `POST /test-sms`

* Send sample SMS using Twilio

# `POST /test-email`

* Send sample email using EmailJS



 How It Works:

1. User sends natural query like "I want amala in Ibadan"
2. API extracts food + city via Groq
3. Coordinates are fetched from Geoapify
4. Restaurants are searched and filtered
5. If no match, fallback via semantic similarity or LLaMA suggestion
6. Result is saved + returned with nearby matches


 Fallback Intelligence:

* If no restaurants are found:

  * Check for semantically similar foods
  * Try alternative suggestions from LLaMA



 Project Structure:

```
.
├── main.py               # Core FastAPI app
├── db_utils.py          # DB helpers (users, history)
├── start.sh             # Startup script (Ollama + FastAPI)
├── Dockerfile           # Docker build
├── render.yaml          # Render deploy settings
├── food_scout_client.py # Optional CLI test client
├── requirements.txt     # Dependencies
├── runtime.txt          # Python version
├── food_scout.db        # SQLite DB (auto-created)
```



 Getting Started (Local):

### 1. Clone & Set Up

```bash
git clone https://github.com/yourname/food-scout-ai
cd food-scout-ai
```

### 2. Environment

Create `.env` with:

```env
GROQ_API_KEY=...
GEOAPIFY_API_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...
```

### 3. Run Locally

```bash
bash start.sh
```

> Visit: [http://localhost:8000/docs](http://localhost:8000/docs) to test API

---

LLM Prompts Used in the System (for Groq / LLaMA3): 

 * Extract food and city from user input:
   Function: extract_food_and_location_groq():

   Prompt logic:
   prompt = f"""
Extract the food and city from this sentence: "{user_input}".
Return only this exact JSON format:
{{ "food": "...", "city": "..." }}
"""

System instruction:
You are a helpful assistant that extracts food and city names from user input. Return only valid JSON like { \"food\": \"amala\", \"city\": \"Ibadan\" } with no explanation.





 * Fallback: Suggest a similar Nigerian food:
   Function: get_llama_suggested_food()

   Prompt logic:
   prompt = f"""
The user asked for '{original_food}' but we couldn’t find it in nearby restaurants.
Suggest a similar Nigerian dish they might enjoy instead.
Just return valid JSON like: {{ "suggested_food": "...", "reason": "..." }}
"""




* Infer a food from vague user input:
  Function: infer_food_from_message()

 Prompt logic:
 prompt = f"""
Based on the user's message: "{message}", suggest the most likely Nigerian food they are craving.
Reply ONLY in this JSON format: {{ "inferred_food": "..." }}
"""

 System instruction:
 You are an assistant that suggests Nigerian dishes from vague cravings. Respond ONLY with JSON like { \"inferred_food\": \"pepper soup\" }




##  Deploy to Render

Ensure these env vars are set in `render.yaml` or the Render UI:

* `GROQ_API_KEY`
* `GEOAPIFY_API_KEY`
* `TWILIO_*`

---

##  Contribution Guide

* Fork and clone
* Create feature branches
* Use clear commit messages
* Push + open a PR

---

## 🙏 Acknowledgements

* [Groq + LLaMA](https://groq.com/)
* [Geoapify](https://www.geoapify.com/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Twilio](https://www.twilio.com/)
* [EmailJS](https://www.emailjs.com/)
