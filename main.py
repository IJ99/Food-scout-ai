import os
import json
import re
import sqlite3
import requests
import traceback
import numpy as np

from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from twilio.rest import Client
from sklearn.metrics.pairwise import cosine_similarity

# Local utility imports
from db_utils import get_user_by_email, create_user, save_search, get_last_search, get_all_searches
from database import initialize_memory_table, initialize_food_table

# ============================================================
# App setup
# ============================================================

app = FastAPI(title="Food Scout AI API")

# 🔥 Heavy model globals
embedding_model = None
known_foods = []
known_food_embeddings = None

@app.on_event("startup")
def load_model_once():
    global embedding_model, known_foods, known_food_embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded")

    # init DB tables
    initialize_memory_table()
    initialize_food_table()

    # Load foods from DB
   # Load foods from DB
known_foods = load_known_foods_from_db()

# If DB is empty, seed default foods
if not known_foods:
    print("⚠️ No foods in DB, seeding defaults...")
    default_foods = [
        "Jollof Rice", "Fried Rice", "Coconut Rice", "White Rice & Stew",
        "Beans & Plantain", "Moi Moi", "Akara", "Yam Porridge", "Boiled Yam & Egg Sauce", "Plantain Chips",
        "Amala", "Eba", "Pounded Yam", "Fufu", "Semovita", "Wheat Swallow", "Tuwo Shinkafa", "Lafun", "Starch",
        "Egusi Soup", "Ogbono Soup", "Okra Soup", "Efo Riro", "Afang Soup", "Edikaikong Soup",
        "Bitterleaf Soup", "Nsala Soup", "Banga Soup", "Oha Soup", "Groundnut Soup", "Fisherman Soup", "Pepper Soup",
        "Suya", "Kilishi", "Grilled Fish", "Asun", "Nkwobi", "Isi Ewu", "Peppered Snail", "Chicken & Chips", "Catfish Pepper Soup",
        "Puff Puff", "Meat Pie", "Fish Roll", "Egg Roll", "Gala Sausage Roll", "Chin Chin", "Kuli Kuli", "Boli", "Shawarma",
        "Pap", "Ogi & Akara", "Agege Bread & Akara", "Yam & Egg Sauce", "Custard & Moi Moi",
        "Zobo Drink", "Kunu", "Palm Wine", "Chapman", "Tiger Nut Drink"
    ]
    conn = sqlite3.connect("food_scout.db")
    cur = conn.cursor()
    cur.executemany("INSERT OR IGNORE INTO foods (name) VALUES (?)", [(f,) for f in default_foods])
    conn.commit()
    conn.close()
    known_foods = default_foods

# Now embed foods
known_food_embeddings = embedding_model.encode(known_foods)
print(f"✅ Loaded {len(known_foods)} foods into embeddings")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Models
# ============================================================

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

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    response: str
    inferred_food: Optional[str] = None
    restaurants: Optional[list] = []
    count: Optional[int] = 0

# ============================================================
# Routes
# ============================================================

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

# ============================================================
# Utils
# ============================================================

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

def find_similar_food(query_food):
    global embedding_model, known_food_embeddings
    if embedding_model is None or known_food_embeddings is None:
        raise HTTPException(status_code=500, detail="Embedding model not loaded yet")

    query_vec = embedding_model.encode([query_food])
    similarities = cosine_similarity(query_vec, known_food_embeddings)[0]
    best_match_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_match_idx])
    best_food = known_foods[best_match_idx]

    if best_score > 0.6 and best_food != query_food:
        return best_food
    return None

# ============================================================
# External API helpers (Groq, Geoapify, Twilio, EmailJS)
# ============================================================

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
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
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

def extract_food_and_location_groq(user_input: str):
    prompt = f"""
    You are a helpful assistant that extracts food items and location (city) from user messages. 
    Extract the food and city from the message below. Respond ONLY in JSON:

    {{"food": "food_name", "city": "city_name"}}
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You extract food and city from user input and respond with JSON only."},
            {"role": "user", "content": prompt + f'Message: "{user_input}"'}
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }
    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        return json.loads(match.group(0)) if match else {"food": None, "city": None}
    except Exception as e:
        print("❌ Groq extraction error:", e)
        return {"food": None, "city": None}

def geocode_city(city_name):
    GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
    url = "https://api.geoapify.com/v1/geocode/search"
    params = {"text": city_name, "limit": 1, "apiKey": GEOAPIFY_API_KEY}
    try:
        res = requests.get(url, params=params)
        data = res.json()
        if data.get("features"):
            coords = data["features"][0]["geometry"]["coordinates"]
            return coords[1], coords[0]
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None, None

def extract_food_from_restaurant_name(name):
    name = name.lower()
    return [food for food in known_foods if food in name]

def search_places_nearby(keyword, lat, lon, radius=5000):
    GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
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
        restaurants, new_foods_found = [], []

        for place in data.get("features", []):
            props = place["properties"]
            name = props.get("name", "Unknown")
            matches = extract_food_from_restaurant_name(name)
            new_foods_found.extend(matches)

            restaurants.append({
                "name": name,
                "address": props.get("formatted", "No address"),
                "lat": place["geometry"]["coordinates"][1],
                "lon": place["geometry"]["coordinates"][0]
            })

        if new_foods_found:
            added_any = False
            for food in set(new_foods_found):
                if food not in known_foods:
                    known_foods.append(food)
                    add_food_to_db(food)
                    added_any = True
            if added_any and embedding_model:
                global known_food_embeddings
                known_food_embeddings = embedding_model.encode(known_foods)

        return restaurants
    except Exception as e:
        print(f"Search error: {e}")
        return []
# ============================================================
# Environment
# ============================================================
load_dotenv()  # make sure env vars are available

# ============================================================
# Core endpoints
# ============================================================

@app.post("/extract", response_model=FoodLocationResponse)
def extract_food_location(request: FoodLocationRequest):
    try:
        result = extract_food_and_location_groq(request.user_input)  # {"food": ..., "city": ...}
        ok = bool(result.get("food")) and bool(result.get("city"))
        return FoodLocationResponse(
            food=result.get("food"),
            city=result.get("city"),
            success=ok,
            error=None if ok else "Could not extract food or city",
        )
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
            "count": len(restaurants),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/full-search")
def full_search(request: FoodLocationRequest):
    try:
        # 1) Ensure user exists
        user = get_user_by_email(request.email)
        user_id = user["id"] if user else create_user(request.name, request.email)

        # 2) Extract
        extracted = extract_food_and_location_groq(request.user_input)
        if not extracted.get("food") or not extracted.get("city"):
            raise HTTPException(status_code=400, detail="Could not extract food or city")

        # 3) Geocode
        lat, lon = geocode_city(extracted["city"])
        if not lat or not lon:
            raise HTTPException(status_code=400, detail="City not found")

        # 4) Search
        restaurants = search_places_nearby(extracted["food"], lat, lon)
        message = None

        # 5) Fallbacks
        if not restaurants:
            sim_food = find_similar_food(extracted["food"])
            if sim_food:
                restaurants = search_places_nearby(sim_food, lat, lon)
                if restaurants:
                    extracted["suggested_food"] = sim_food
                    message = (
                        f"No restaurants found for '{extracted['food']}', "
                        f"but here are results for '{sim_food}'."
                    )
            if not restaurants:
                llama = get_llama_suggested_food(extracted["food"])
                if llama and llama.get("suggested_food"):
                    alt = llama["suggested_food"]
                    reason = llama.get("reason", "")
                    restaurants = search_places_nearby(alt, lat, lon)
                    if restaurants:
                        extracted["suggested_food"] = alt
                        message = f"{reason} Here are results for '{alt}' instead."
                if not restaurants:
                    extracted["suggested_food"] = None
                    message = f"No restaurants found for '{extracted['food']}' or close alternatives."

        if "suggested_food" not in extracted:
            extracted["suggested_food"] = None

        # 6) Save history (always save original extracted)
        save_search(user_id, extracted["food"], extracted["city"])

        # 7) Return
        return {
            "user_input": request.user_input,
            "extracted": extracted,
            "coordinates": {"lat": lat, "lon": lon},
            "restaurants": restaurants,
            "count": len(restaurants),
            "message": message,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{email}")
def get_search_history(email: str):
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    history = get_all_searches(email)
    return {
        "user": {"name": user["name"], "email": user["email"]},
        "history": history,
        "count": len(history),
    }


@app.get("/recommend/{email}")
def recommend_foods(email: str):
    global embedding_model, known_food_embeddings
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if embedding_model is None or known_food_embeddings is None or not len(known_food_embeddings):
        raise HTTPException(status_code=500, detail="Embedding model or food embeddings not ready")

    history = get_all_searches(email)
    past_foods = list({entry["food"] for entry in history if entry.get("food")})
    if not past_foods:
        return {"message": "Not enough history for recommendations yet.", "recommendations": []}

    past_embeddings = embedding_model.encode(past_foods)
    mean_vec = np.mean(past_embeddings, axis=0).reshape(1, -1)

    similarities = cosine_similarity(mean_vec, known_food_embeddings)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    recommendations = []
    for idx in ranked_indices:
        food = known_foods[idx]
        if food not in past_foods:
            recommendations.append(food)
        if len(recommendations) >= 3:
            break

    return {
        "user": {"name": user["name"], "email": user["email"]},
        "history_count": len(past_foods),
        "past_foods": past_foods,
        "recommendations": recommendations,
    }

# ============================================================
# Chat + memory
# ============================================================

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
    cur.execute(
        "INSERT INTO conversation_memory (session_id, role, message) VALUES (?, ?, ?)",
        (session_id, role, message),
    )
    conn.commit()
    conn.close()

def get_conversation(session_id):
    conn = sqlite3.connect("food_scout.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT role, message FROM conversation_memory WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"role": role, "content": message} for role, message in rows]

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        memory = get_conversation(request.session_id)
        memory.append({"role": "user", "content": request.message})

        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={"model": "llama3-8b-8192", "messages": memory, "temperature": 0.6},
            timeout=30,
        )
        content = res.json()["choices"][0]["message"]["content"]

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
    prompt = (
        f'Based on the user message: "{message}", '
        'suggest the most likely Nigerian food they are craving. '
        'Reply ONLY as JSON: {"inferred_food": "..."}'
    )
    try:
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            'You suggest Nigerian dishes from vague cravings. '
                            'Respond ONLY with JSON like {"inferred_food":"pepper soup"}.'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=30,
        )
        content = res.json()["choices"][0]["message"]["content"]
        match = re.search(r"\{.*?\}", content, re.DOTALL)
        return (json.loads(match.group(0))["inferred_food"] if match else None)
    except Exception as e:
        print("⚠️ Inference failed:", e)
        return None

@app.post("/chat-smart", response_model=ChatResponse)
def chat_smart(request: ChatRequest):
    try:
        memory = get_conversation(request.session_id)
        memory.append({"role": "user", "content": request.message})

        all_text = " ".join([m["content"] for m in memory]).lower()
        past_foods = [f for f in known_foods if f in all_text]

        inferred = None
        if not past_foods:
            inferred = infer_food_from_message(request.message)
            if inferred:
                past_foods = [inferred]

        context = ", ".join(past_foods) if past_foods else "No known foods mentioned yet."
        system_msg = {
            "role": "system",
            "content": f"You are a food assistant. Previously mentioned foods: {context}. Be personal and helpful.",
        }

        full_memory = [system_msg] + memory
        res = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={"model": "llama3-8b-8192", "messages": full_memory, "temperature": 0.6},
            timeout=30,
        )
        content = res.json()["choices"][0]["message"]["content"]

        save_message(request.session_id, "user", request.message)
        save_message(request.session_id, "assistant", content)

        restaurants = []
        if inferred:
            lat, lon = geocode_city("Lagos")  # TODO: make location dynamic
            if lat and lon:
                restaurants = search_places_nearby(inferred, lat, lon)

        return ChatResponse(
            session_id=request.session_id,
            response=content,
            inferred_food=inferred,
            restaurants=restaurants,
            count=len(restaurants),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summary/{session_id}")
def summarize_session(session_id: str):
    try:
        memory = get_conversation(session_id)
        text = " ".join([m["content"].lower() for m in memory])
        mentioned = list({food for food in known_foods if food in text})
        return {"session_id": session_id, "mentioned_foods": mentioned, "count": len(mentioned)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Twilio + Email test endpoints
# ============================================================

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

@app.post("/test-sms")
def test_sms():
    if not twilio_client or not TWILIO_PHONE_NUMBER:
        raise HTTPException(status_code=400, detail="Twilio not configured")
    try:
        msg = twilio_client.messages.create(
            body="Hello from AI Food Scout 🧠🍽️",
            from_=TWILIO_PHONE_NUMBER,
            to="+2348074261515",  # replace with a verified number
        )
        print("✅ SMS sent:", msg.sid)
        return {"success": True}
    except Exception as e:
        print("❌ Failed to send SMS:", e)
        return {"success": False, "error": str(e)}

def send_email_via_emailjs(to_email, food, restaurant):
    EMAILJS_SERVICE_ID = os.getenv("EMAILJS_SERVICE_ID")
    EMAILJS_TEMPLATE_ID = os.getenv("EMAILJS_TEMPLATE_ID")
    EMAILJS_PUBLIC_KEY = os.getenv("EMAILJS_PUBLIC_KEY")

    if not (EMAILJS_SERVICE_ID and EMAILJS_TEMPLATE_ID and EMAILJS_PUBLIC_KEY):
        print("⚠️ EmailJS not configured")
        return False

    payload = {
        "service_id": EMAILJS_SERVICE_ID,
        "template_id": EMAILJS_TEMPLATE_ID,
        "user_id": EMAILJS_PUBLIC_KEY,
        "template_params": {"email": to_email, "food": food, "restaurant": restaurant},
    }
    try:
        r = requests.post(
            "https://api.emailjs.com/api/v1.0/email/send",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        ok = r.status_code == 200
        print("✅ Email sent" if ok else f"❌ Email failed: {r.text}")
        return ok
    except Exception as e:
        print("⚠️ Error sending email:", e)
        return False

@app.post("/test-email")
def test_email():
    ok = send_email_via_emailjs(
        to_email="you@example.com",
        food="Grilled Suya",
        restaurant="Mama Nkechi Kitchen",
    )
    return {"success": ok}
