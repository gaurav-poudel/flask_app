from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import joblib
import numpy as np
import re
import spacy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)  

# Global variables to hold models
recommendation_model = None
intent_classifier = None
response_templates = None
nlp = None

# Global state dictionaries for chatbot
conversation_states = {}
pending_bookings = {}
new_booking_dates = {}
new_booking_times = {}
booking_services = {}
booking_database = {}

def load_models():
    """Load recommendation and chatbot models"""
    global recommendation_model, intent_classifier, response_templates, nlp
    
    try:
        print("Loading spaCy model...")
        nlp = spacy.load('en_core_web_sm')
        print("spaCy model loaded successfully")
    except Exception as e:
        print(f"Error loading spaCy model: {e}")
        nlp = None
    
    try:
        print("Loading recommendation model...")
        recommendation_model = joblib.load('recommendation_model.pkl')
        print("Recommendation model loaded successfully")
    except Exception as e:
        print(f"Error loading recommendation model: {e}")
        recommendation_model = None
    
    try:
        print("Loading chatbot model...")
        with open('chatbot_model.pkl', 'rb') as f:
            chatbot_data = pickle.load(f)
            intent_classifier = chatbot_data['intent_classifier']
            response_templates = chatbot_data['response_templates']
        print("Chatbot model loaded successfully")
    except Exception as e:
        print(f"Error loading chatbot model: {e}")
        intent_classifier = None
        response_templates = None

# Helper functions for the chatbot
def get_random_response(intent):
    """Get a random response template for the given intent"""
    templates = response_templates.get(intent, response_templates.get("default", ["I'm not sure how to respond to that."]))
    return np.random.choice(templates)

def predict_intent(text):
    """Predict the intent of user input"""
    try:
        intent_probs = intent_classifier.predict_proba([text])[0]
        intent_idx = np.argmax(intent_probs)
        confidence = intent_probs[intent_idx]
        
        if confidence < 0.4:  
            return "default"
        
        return intent_classifier.classes_[intent_idx]
    except Exception as e:
        print(f"Error predicting intent: {e}")
        return "default"

def extract_date_time(text):
    """Extract date and time from user input"""
    if nlp is None:
        return None, None
        
    doc = nlp(text)
    
    date_pattern = r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b'
    time_pattern = r'\b(\d{1,2})(:\d{2})?\s*(am|pm)\b'
    
    date_match = re.search(date_pattern, text, re.IGNORECASE)
    time_match = re.search(time_pattern, text, re.IGNORECASE)
    
    date_str = None
    time_str = None
    
    if date_match:
        date_str = date_match.group(0)
    
    if time_match:
        time_str = time_match.group(0)
        
    dates = []
    times = []
    
    for ent in doc.ents:
        if ent.label_ == "DATE":
            dates.append(ent.text)
        elif ent.label_ == "TIME":
            times.append(ent.text)
    
    if date_str is None and dates:
        date_str = dates[0]
    
    if time_str is None and times:
        time_str = times[0]
        
    return date_str, time_str

def update_booking(booking_id, new_date, new_time):
    """Update a booking with new date and time"""
    if booking_id in booking_database:
        booking_database[booking_id]["date"] = new_date
        booking_database[booking_id]["time"] = new_time
        return True
    return False

def create_booking(service, date, time):
    """Create a new booking"""
    booking_id = len(booking_database) + 1001
    booking_database[booking_id] = {
        "service": service,
        "date": date,
        "time": time,
        "status": "confirmed"
    }
    return booking_id

def process_message(message, session_id="default"):
    """Process user message and return appropriate response"""
    if intent_classifier is None or response_templates is None:
        return "Sorry, the chatbot is currently unavailable."
    
    if session_id not in conversation_states:
        conversation_states[session_id] = "start"
        pending_bookings[session_id] = None
        new_booking_dates[session_id] = None
        new_booking_times[session_id] = None
        booking_services[session_id] = None
    
    current_state = conversation_states[session_id]
    
    if current_state == "start":
        intent = predict_intent(message)
        
        if intent == "rescheduling":
            conversation_states[session_id] = "confirm_reschedule"
            pending_bookings[session_id] = 1001  
            return get_random_response("rescheduling")
            
        elif intent == "booking":
            conversation_states[session_id] = "ask_service"
            return get_random_response("booking")
            
        else:
            return get_random_response(intent)
            
    elif current_state == "confirm_reschedule":
        lower_msg = message.lower()
        if "yes" in lower_msg or "sure" in lower_msg or "okay" in lower_msg:
            conversation_states[session_id] = "ask_reschedule_date"
            return get_random_response("rescheduling_confirmation")
        else:
            conversation_states[session_id] = "start"
            return "I'll keep your booking as is. Is there anything else I can help with?"
            
    elif current_state == "ask_reschedule_date":
        date_str, time_str = extract_date_time(message)
        
        if date_str and time_str:
            new_booking_dates[session_id] = date_str
            new_booking_times[session_id] = time_str
            
            success = update_booking(pending_bookings[session_id], date_str, time_str)
            
            conversation_states[session_id] = "start"
            return get_random_response("rescheduling_success")
            
        elif date_str:
            new_booking_dates[session_id] = date_str
            conversation_states[session_id] = "ask_reschedule_time"
            return f"Got it, {date_str}. What time would you prefer?"
            
        else:
            return "I couldn't understand the date. Please provide a date in format like '30 Mar 2025'."
            
    elif current_state == "ask_reschedule_time":
        _, time_str = extract_date_time(message)
        
        if time_str:
            new_booking_times[session_id] = time_str
            
            success = update_booking(
                pending_bookings[session_id], 
                new_booking_dates[session_id], 
                time_str
            )
            
            conversation_states[session_id] = "start"
            return get_random_response("rescheduling_success")
        else:
            return "I couldn't understand the time. Please provide a time in format like '10 am' or '2:30 pm'."
            
    elif current_state == "ask_service":
        booking_services[session_id] = message
        conversation_states[session_id] = "ask_booking_date"
        return f"Great choice! When would you like to book the {message}?"
        
    elif current_state == "ask_booking_date":
        date_str, time_str = extract_date_time(message)
        
        if date_str and time_str:
            booking_id = create_booking(
                booking_services[session_id], 
                date_str, 
                time_str
            )
            
            conversation_states[session_id] = "start"
            return f"Your booking for {booking_services[session_id]} on {date_str} at {time_str} is confirmed! Your booking reference is #{booking_id}."
            
        elif date_str:
            new_booking_dates[session_id] = date_str
            conversation_states[session_id] = "ask_booking_time"
            return f"Got it, {date_str}. What time would you prefer?"
            
        else:
            return "I couldn't understand the date. Please provide a date in format like '30 Mar 2025'."
            
    elif current_state == "ask_booking_time":
        _, time_str = extract_date_time(message)
        
        if time_str:
            booking_id = create_booking(
                booking_services[session_id], 
                new_booking_dates[session_id], 
                time_str
            )
            
            conversation_states[session_id] = "start"
            return f"Your booking for {booking_services[session_id]} on {new_booking_dates[session_id]} at {time_str} is confirmed! Your booking reference is #{booking_id}."
        else:
            return "I couldn't understand the time. Please provide a time in format like '10 am' or '2:30 pm'."
    
    return "I'm not sure I understand. Can you please try again?"

def recommend_services(customer_id, top_n=3):
    """Get service recommendations for a specific customer"""
    if recommendation_model is None:
        return {"error": "Recommendation model not loaded"}
        
    try:
        service_similarity_df = recommendation_model.get('service_similarity')
        if service_similarity_df is None:
            return {"error": "Service similarity matrix not found in model"}
            
        customer_id = int(customer_id)
        
        customer_data = {}
        
        if 'customer_data' in recommendation_model:
            customer_data = recommendation_model['customer_data']
        
        if customer_data and customer_id in customer_data:
            preferred_service = customer_data[customer_id].get('preferred_service')
        else:
            try:
                if hasattr(recommendation_model, 'recommend_services'):
                    return recommendation_model.recommend_services(customer_id, top_n)
                else:
                    return {"error": "Customer not found and no recommendation function available"}
            except Exception as e:
                return {"error": f"Error using recommendation function: {str(e)}"}
        
        if preferred_service not in service_similarity_df.index:
            return {"error": f"Service '{preferred_service}' not found in similarity matrix"}
            
        similar_services = service_similarity_df.loc[preferred_service].sort_values(ascending=False)
        similar_services = similar_services.drop(preferred_service).head(top_n)
        
        recommendations = []
        for service, score in similar_services.items():
            recommendations.append({
                "service": service,
                "score": round(float(score), 2)
            })
        
        return {
            "customer_id": customer_id,
            "current_service": preferred_service,
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": f"Error generating recommendations: {str(e)}"}

# Define API routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        "status": "API is running",
        "endpoints": {
            "/api/recommend": "POST - Get service recommendations for a customer",
            "/api/chatbot": "POST - Interact with the chatbot"
        }
    })

@app.route('/chat', methods=['GET'])
def chat_interface():
    return render_template('chat.html')

@app.route('/api/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        # Handle GET request (for browser testing)
        customer_id = request.args.get('customer_id', 1001)  # Default to 1001 if not provided
        result = recommend_services(customer_id)
        return jsonify(result)
    
    # Existing POST handler
    data = request.get_json()
    if not data or 'customer_id' not in data:
        return jsonify({"error": "Missing customer_id parameter"}), 400
    
    customer_id = data.get('customer_id')
    top_n = data.get('top_n', 3)
    
    result = recommend_services(customer_id, top_n)
    
    if 'error' in result:
        return jsonify(result), 404
    
    return jsonify(result)

@app.route('/api/debug-model', methods=['GET'])
def debug_model():
    if recommendation_model is None:
        return jsonify({"error": "Model not loaded"})
    
    model_info = {
        "type": str(type(recommendation_model)),
        "keys": list(recommendation_model.keys()) if isinstance(recommendation_model, dict) else "Not a dictionary",
        "has_customer_data": 'customer_data' in recommendation_model if isinstance(recommendation_model, dict) else False,
        "has_service_similarity": 'service_similarity' in recommendation_model if isinstance(recommendation_model, dict) else False
    }
    
    return jsonify(model_info)

@app.route('/api/chatbot', methods=['POST'])
def chat():
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "Missing message parameter"}), 400
    
    message = data.get('message')
    session_id = data.get('session_id', 'default')  # Use default session ID if not provided
    
    response = process_message(message, session_id)
    
    return jsonify({
        "response": response,
        "session_id": session_id
    })

# Check if templates directory exists and create it if needed
def setup_templates():
    if not os.path.exists('templates'):
        os.makedirs('templates')

# Load models and set up templates on startup
load_models()
setup_templates()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5051)