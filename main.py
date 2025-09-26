import os
import json
import re
import sqlite3
import requests

from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer 

# Local utility imports
from db_utils import get_user_by_email, create_user, save_search, get_last_search, get_all_searches

from fastapi import FastAPI, , HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. Create the app FIRST
app = FastAPI()

# 2. Then add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Now define your routes
@app.get("/", include_in_schema=False)
def root():
    return {"status": "Food Scout AI is running!"}

@app.get("/health", include_in_schema=False)
def health_check():
    return {
        "status": "healthy",
        "groq_model": "llama3-8b-8192",
        "api": "online"
    }

# ✅ Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
print("🔑 GROQ KEY LOADED:", groq_key)  # Optional: remove after confirming it works

# ✅ Initialize FastAPI app
app = FastAPI()

import traceback

@app.post("/full-search")
async def full_search(request: FoodLocationRequest):
    try:
        # your existing logic
        ...
    except Exception as e:
        print("❌ ERROR in /full-search:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
        

# ✅ Twilio for notifications
from twilio.rest import Client

def send_sms_notification(restaurant_name, food_item, phone_number):
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_number = os.getenv("TWILIO_PHONE_NUMBER")

        client = Client(account_sid, auth_token)

        message_body = (
            f"🍽️ Hello {restaurant_name}!\n"
            f"A customer is interested in ordering '{food_item}' via AI FoodScout.\n"
            f"Please expect a call or walk-in soon!"
        )

        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=phone_number  # 📌 Must be in +234... format
        )

        print("✅ SMS sent:", message.sid)
        return True

    except Exception as e:
        print("❌ Failed to send SMS:", e)
        return False


def load_known_foods_from_db():
    conn = sqlite3.connect("food_scout.db")
    cur = conn.cursor()
    cur.execute("SELECT name FROM foods")
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from database import initialize_memory_table
initialize_memory_table()

from database import initialize_food_table
initialize_food_table()


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Food Scout AI API")

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class FoodLocationRequest(BaseModel):
    user_input: str
    name: str
    email: str

class FoodLocationResponse(BaseModel):
    food: Optional[str]
    city: Optional[str]
    success: bool
    error: Optional[str] = None

class RestaurantSearchRequest(BaseModel):
    food: str
    city: str
    radius: int = 5000
    
def load_known_foods_from_db():
    conn = sqlite3.connect("food_scout.db")
    cur = conn.cursor()
    cur.execute("SELECT name FROM foods")
    rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]
   

def add_food_to_db(food):
    conn = sqlite3.connect("food_scout.db")
    cur = conn.cursor()
    try:
        cur.execute("INSERT OR IGNORE INTO foods (name) VALUES (?)", (food,))
        conn.commit()
    finally:
        conn.close()

# ✅ Load known foods from DB and encode them
known_foods = load_known_foods_from_db()
known_food_embeddings = embedding_model.encode(known_foods)


# ✅ Similar food finder
def find_similar_food(query_food):
    query_vec = embedding_model.encode([query_food])
    similarities = cosine_similarity(query_vec, known_food_embeddings)[0]

    best_match_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_match_idx])
    best_food = known_foods[best_match_idx]

    if best_score > 0.6 and best_food != query_food:
        return best_food
    else:
        return None


# 🔥 Real-time fallback using LLaMA (if no close match found)
def get_llama_suggested_food(original_food):
    prompt = f"""
    The user asked for '{original_food}' but we couldn’t find it in nearby restaurants.
    Suggest a similar Nigerian dish they might enjoy instead.
    Just return valid JSON like: {{ "suggested_food": "...", "reason": "..." }}
    """

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4
            },
            timeout=30
        )
        output = response.json()["choices"][0]["message"]["content"]
        match = re.search(r'\{.*?\}', output, re.DOTALL)
        return json.loads(match.group(0))
    except Exception as e:
        print("⚠️ LLaMA suggestion failed:", e)
        return None

# --- Extract food & location using Groq ---
def extract_food_and_location_groq(user_input: str):
    prompt = f"""
You are a helpful assistant that extracts food items and location (city) from user messages. 
Extract the food and city from the message below. Respond ONLY in the following JSON format:

{{"food": "food_name", "city": "city_name"}}

If food or city is not found, return null for that field.

Message: "{user_input}"
"""

    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {"role": "system", "content": "You extract food and city from user input and respond with JSON only."},
        {"role": "user", "content": prompt}
    ],
    "temperature": 0,
    "response_format": { "type": "json_object" }
}

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]

        print("🧠 LLaMA response:", content)  # For debugging

        # Force JSON extraction
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            json_part = match.group(0)
            result = json.loads(json_part)
            return {
                "food": result.get("food"),
                "city": result.get("city")
            }

        return {"food": None, "city": None}

    except Exception as e:
        print("❌ Groq extraction error:", e)
        return {"food": None, "city": None}


# --- Geocoding ---
def geocode_city(city_name):
    GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "3f1325af262f410fb28f00e9ae4bbcb2")
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {"text": city_name, "limit": 1, "apiKey": GEOAPIFY_API_KEY}
    try:
        res = requests.get(url, params=params)
        data = res.json()
        if data.get("features"):
            coords = data["features"][0]["geometry"]["coordinates"]
            return coords[1], coords[0]
        else:
            return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None
# ✅ Auto-extract food from restaurant names
def extract_food_from_restaurant_name(name):
    name = name.lower()
    matched = [food for food in known_foods if food in name]
    return matched  # return list (can be empty)

# --- Restaurant search ---
def search_places_nearby(keyword, lat, lon, radius=5000):
    GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY", "3f1325af262f410fb28f00e9ae4bbcb2")
    url = "https://api.geoapify.com/v2/places"
    params = {
        "categories": "catering.restaurant",
        "filter": f"circle:{lon},{lat},{radius}",
        "text": keyword,
        "bias": f"proximity:{lon},{lat}",
        "limit": 10,
        "apiKey": GEOAPIFY_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        restaurants = []
        new_foods_found = []

        for place in data.get("features", []):
            props = place["properties"]
            name = props.get("name", "Unknown")

            # 🔍 Extract food names from restaurant name
            matches = extract_food_from_restaurant_name(name)
            new_foods_found.extend(matches)

            restaurants.append({
                "name": name,
                "address": props.get("formatted", "No address"),
                "lat": place["geometry"]["coordinates"][1],
                "lon": place["geometry"]["coordinates"][0]
            })

        # 🔁 Add new food items to known_foods list
        if new_foods_found:
            added_any = False
            for food in set(new_foods_found):
                if food not in known_foods:
                    known_foods.append(food)
                    add_food_to_db(food)  # ✅ Persist to DB
                    print(f"🍽️ Auto-added new food: {food}")
                    added_any = True
            if added_any:
                # 🧠 Recalculate embeddings
                global known_food_embeddings
                known_food_embeddings = embedding_model.encode(known_foods)

        return restaurants
    except Exception as e:
        print(f"Search error: {e}")
        return []
    
@app.get("/")
def root():
    return {"message": "Food Scout AI API is running!", "status": "healthy"}

@app.post("/extract", response_model=FoodLocationResponse)
def extract_food_location(request: FoodLocationRequest):
    try:
        result = extract_food_and_location_groq(request.user_input)
        if result["food"] and result["city"]:
            return FoodLocationResponse(food=result["food"], city=result["city"], success=True)
        else:
            return FoodLocationResponse(food=result.get("food", ""), city=result.get("city", ""), success=False, error="Could not extract food or city")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-restaurants")
def search_restaurants(request: RestaurantSearchRequest):
    try:
        lat, lon = geocode_city(request.city)
        if not lat or not lon:
            raise HTTPException(status_code=400, detail="City not found")
        restaurants = search_places_nearby(request.food, lat, lon, request.radius)
        return {
            "food": request.food,
            "city": request.city,
            "coordinates": {"lat": lat, "lon": lon},
            "restaurants": restaurants,
            "count": len(restaurants)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/full-search")
def full_search(request: FoodLocationRequest):
    try:
        # 1. Check or create user
        user = get_user_by_email(request.email)
        if not user:
            user_id = create_user(request.name, request.email)
        else:
            user_id = user["id"]

        # 2. Extract food and city
        extracted = extract_food_and_location_groq(request.user_input)
        if not extracted["food"] or not extracted["city"]:
            raise HTTPException(status_code=400, detail="Could not extract food or city")

        # 3. Geocode city
        lat, lon = geocode_city(extracted["city"])
        if not lat or not lon:
            raise HTTPException(status_code=400, detail="City not found")

        # 4. Search restaurants
        restaurants = search_places_nearby(extracted["food"], lat, lon)

        # 5. Fallback logic if no results
        message = None
        if not restaurants:
            suggested_food = find_similar_food(extracted["food"])
            
            if suggested_food:
                print(f"🔁 Retrying with similar food: {suggested_food}")
                restaurants = search_places_nearby(suggested_food, lat, lon)
                if restaurants:
                    extracted["suggested_food"] = suggested_food
                    message = (
                        f"No restaurants found for '{extracted['food']}', "
                        f"but here are some results for '{suggested_food}'."
                    )
                else:
                    # 🧠 Try LLaMA suggestion
                    llama_suggestion = get_llama_suggested_food(extracted["food"])
                    if llama_suggestion:
                        alt_food = llama_suggestion.get("suggested_food")
                        reason = llama_suggestion.get("reason", "")
                        print(f"🧠 LLaMA suggested: {alt_food} — {reason}")

                        restaurants = search_places_nearby(alt_food, lat, lon)
                        if restaurants:
                            extracted["suggested_food"] = alt_food
                            message = f"{reason} Here are results for '{alt_food}' instead."
                        else:
                            extracted["suggested_food"] = None
                            message = f"No restaurants found for '{extracted['food']}' or suggested alternatives."
                    else:
                        extracted["suggested_food"] = None
                        message = f"No restaurants found for '{extracted['food']}' or any close match."

            else:
                # Directly try LLaMA suggestion
                llama_suggestion = get_llama_suggested_food(extracted["food"])
                if llama_suggestion:
                    alt_food = llama_suggestion.get("suggested_food")
                    reason = llama_suggestion.get("reason", "")
                    print(f"🧠 LLaMA fallback: {alt_food} — {reason}")

                    restaurants = search_places_nearby(alt_food, lat, lon)
                    if restaurants:
                        extracted["suggested_food"] = alt_food
                        message = f"{reason} Here are results for '{alt_food}' instead."
                    else:
                        extracted["suggested_food"] = None
                        message = f"No restaurants found for '{extracted['food']}' or suggested alternatives."
                else:
                    extracted["suggested_food"] = None
                    message = f"No restaurants found for '{extracted['food']}' and no similar food could be suggested."

        else:
            extracted["suggested_food"] = None  # No suggestion needed

        # 6. Save original food + city
        print(f"📥 Saving search for user_id={user_id}, food={extracted['food']}, city={extracted['city']}")
        save_search(user_id, extracted["food"], extracted["city"])

        # 7. Return response
        return {
            "user_input": request.user_input,
            "extracted": extracted,
            "coordinates": {"lat": lat, "lon": lon},
            "restaurants": restaurants,
            "count": len(restaurants),
            "message": message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "groq_model": "llama3-8b-8192",
        "api": "online"
    }

@app.get("/history/{email}")
def get_search_history(email: str):
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    history = get_all_searches(email)
    return {
        "user": {
            "name": user["name"],
            "email": user["email"]
        },
        "history": history,
        "count": len(history)
    } 

@app.get("/recommend/{email}")
def recommend_foods(email: str):
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get food history
    history = get_all_searches(email)
    past_foods = list({entry["food"] for entry in history})  # Unique foods

    if not past_foods:
        return {
            "message": "You don't have enough history for recommendations yet.",
            "recommendations": []
        }

    # Embed the user's food history
    past_embeddings = embedding_model.encode(past_foods)
    mean_vec = np.mean(past_embeddings, axis=0).reshape(1, -1)

    # Compare mean vector to known food embeddings
    similarities = cosine_similarity(mean_vec, known_food_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    # Recommend top 3 foods they haven't searched
    recommendations = []
    for idx in ranked_indices:
        food = known_foods[idx]
        if food not in past_foods:
            recommendations.append(food)
        if len(recommendations) >= 3:
            break

    return {
        "user": {
            "name": user["name"],
            "email": user["email"]
        },
        "history_count": len(past_foods),
        "past_foods": past_foods,
        "recommendations": recommendations
    }

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    inferred_food: Optional[str] = None
    restaurants: Optional[list] = []
    count: Optional[int] = 0


def save_message(session_id, role, message):
    conn = sqlite3.connect("food_scout.db")
    cur = conn.cursor()
    cur.execute("INSERT INTO conversation_memory (session_id, role, message) VALUES (?, ?, ?)", (session_id, role, message))
    conn.commit()
    conn.close()

def get_conversation(session_id):
    conn = sqlite3.connect("food_scout.db")
    cur = conn.cursor()
    cur.execute("SELECT role, message FROM conversation_memory WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": role, "content": message} for role, message in rows]

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Load previous messages
        memory = get_conversation(request.session_id)

        # Add new user message
        memory.append({"role": "user", "content": request.message})

        # Call LLaMA with memory
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": memory,
                "temperature": 0.6
            },
            timeout=30
        )
        content = response.json()["choices"][0]["message"]["content"]

        # Save to DB memory
        save_message(request.session_id, "user", request.message)
        save_message(request.session_id, "assistant", content)

        return ChatResponse(session_id=request.session_id, response=content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/{session_id}")
def get_memory(session_id: str):
    try:
        return get_conversation(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def infer_food_from_message(message):
    prompt = f"""
    Based on the user's message: "{message}", suggest the most likely Nigerian food they are craving.
    Reply ONLY in this JSON format: {{ "inferred_food": "..." }}
    """

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an assistant that suggests Nigerian dishes from vague cravings. Respond ONLY with JSON like { \"inferred_food\": \"pepper soup\" }"
                    },
                    { "role": "user", "content": prompt }
                ],
                "temperature": 0.3
            },
            timeout=30
        )

        output = response.json()["choices"][0]["message"]["content"]
        match = re.search(r'\{.*?\}', output, re.DOTALL)
        if match:
            return json.loads(match.group(0)).get("inferred_food")
        else:
            return None

    except Exception as e:
        print("⚠️ Inference failed:", e)
        return None


@app.post("/chat-smart", response_model=ChatResponse)
def chat_smart(request: ChatRequest):
    try:
        memory = get_conversation(request.session_id)
        memory.append({"role": "user", "content": request.message})

        # 🧠 Extract past foods from memory
        all_text = " ".join([msg["content"] for msg in memory])
        past_foods = [food for food in known_foods if food in all_text.lower()]
        # 🧠 Try inference if no foods detected
        inferred = None
        if not past_foods:
             inferred = infer_food_from_message(request.message)
             if inferred:
                 past_foods = [inferred]

        context = ", ".join(past_foods) if past_foods else "No known foods mentioned yet."

        system_message = {
            "role": "system",
            "content": f"You are a food assistant. Previously mentioned foods include: {context}. Give personalized suggestions."
        }

        full_memory = [system_message] + memory

        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": full_memory,
                "temperature": 0.6
            },
            timeout=30
        )
        content = res.json()["choices"][0]["message"]["content"]
        # Save memory
        save_message(request.session_id, "user", request.message)
        save_message(request.session_id, "assistant", content)

# 🧠 Try to infer food
        inferred = infer_food_from_message(request.message)
        restaurants = []
        location = "Lagos"  # You can make this dynamic later

        if inferred:
            print(f"🔍 Inferred food: {inferred}")
            lat, lon = geocode_city(location)
            if lat and lon:
                restaurants = search_places_nearby(inferred, lat, lon)

# 📦 Return everything
        return {
            "session_id": request.session_id,
            "response": content,
            "inferred_food": inferred,
            "restaurants": restaurants,
            "count": len(restaurants)
}


        return ChatResponse(session_id=request.session_id, response=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{session_id}")
def summarize_session(session_id: str):
    try:
        memory = get_conversation(session_id)
        text = " ".join([msg["content"].lower() for msg in memory])
        mentioned = list({food for food in known_foods if food in text})
        return {
            "session_id": session_id,
            "mentioned_foods": mentioned,
            "count": len(mentioned)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import APIRouter
from twilio.rest import Client

# ✅ Load Twilio credentials from environment
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# ✅ Initialize Twilio Client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.post("/test-sms")
def test_sms():
    try:
        message = twilio_client.messages.create(
            body="Hello from AI Food Scout 🧠🍽️",
            from_=+13167516538,
            to="+2348074261515"  # ✅ Your verified number
        )
        print("✅ SMS sent:", message.sid)
        return {"success": True}
    except Exception as e:
        print("❌ Failed to send SMS:", e)
        return {"success": False}

import requests

def send_email_via_emailjs(to_email, food, restaurant):
    EMAILJS_SERVICE_ID = "service_suvj2zs"
    EMAILJS_TEMPLATE_ID = "template_fp7gam9"
    EMAILJS_PUBLIC_KEY = "awZHT6j6u0Ou8-TuB"  # aka USER ID

    payload = {
        "service_id": EMAILJS_SERVICE_ID,
        "template_id": EMAILJS_TEMPLATE_ID,
        "user_id": EMAILJS_PUBLIC_KEY,
        "template_params": {
            "email": to_email,
            "food": food,
            "restaurant": restaurant
        }
    }

    try:
        response = requests.post(
            "https://api.emailjs.com/api/v1.0/email/send",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Email sent successfully")
            return True
        else:
            print("❌ Email failed:", response.text)
            return False
    except Exception as e:
        print("⚠️ Error sending email:", e)
        return False

@app.post("/test-email")
def test_email():
    result = send_email_via_emailjs(
        to_email="ichibor3458@student.babcock.edu.ng",  # replace with YOUR email to receive the test
        food="Grilled Suya",
        restaurant="Mama Nkechi Kitchen"
    )
    return {"success": result}




 

   
    
 
              
              



    
              
              
        
      
       

                 
                          
                   
       


   
  
   
     

   
    
        
    
    

          




   
         
