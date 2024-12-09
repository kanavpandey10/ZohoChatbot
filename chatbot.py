import openai
from flask import Flask, request, jsonify, send_from_directory
import json
import re
import spacy
import uuid
import psycopg2
from psycopg2 import sql, extras
from psycopg2.pool import SimpleConnectionPool

conn = psycopg2.connect(
    dbname="ZohoQuiz",
    user="postgres",
    password="sql2023",
    host="localhost",
    port="5432"
)
app = Flask(__name__)
# Load JSON data
with open('data.json', 'r') as f:
    services_data = json.load(f)
# Set up OpenAI API key
openai.api_key = 'enter your api key'

DATABASE_URL = 'postgresql://postgres:sql2023@localhost:5432/ZohoQuiz'

# Initialize database connection pool
try:
    db_pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
except Exception as e:
    raise Exception(f"Error connecting to database: {e}")

# Pre-load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load JSON data
with open('data.json', 'r') as f:
    services_data = json.load(f)

# Ensure table creation


def create_tables():
    with db_pool.getconn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation (
                id SERIAL PRIMARY KEY,
                session_id UUID NOT NULL,
                user_message TEXT NOT NULL,
                bot_response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            conn.commit()


create_tables()

# Extract entities with spaCy


def extract_entities(message):
    doc = nlp(message)
    entities = {}
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":  # Extract numbers
            entities["quantity"] = ent.text
        elif ent.label_ in {"ORG", "PRODUCT", "GPE"}:  # Extract service names
            entities["service_or_feature"] = ent.text
    return entities


# Manage session data
sessions = {}


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {"history": []}
    return sessions[session_id]

# Call OpenAI for suggestions
def call_gpt_function_suggestion(user_message, entities, history, usage_details, comparison_invoices=None):

    """
    Generate a suggestion or comparison of service plan based on user input
    :param user_message: The user's query.
    :param entities: Extracted entities from the user's message.
    :param history: The conversation history.
    :param usage_details: User's current usage data for generating suggestions
    :param comparison_invoices: Optional. List of additional invoices to compare.
    :return: Chatbot's response.
    """
    history_text = "\n".join(
        [f"User: {h['user']}\nBot: {h['bot']}" for h in history])
    
    # Prepare invoice comparison prompt if comparison invoices are provided
    comparison_prompt = ""
    if comparison_invoices:
        comparison_text = "\n\n".join([f"Invoice {i + 1}:\n{json.dumps(invoice, indent=2)}" for i, invoice in enumerate(comparison_invoices)])
        comparison_prompt = f"""
The user provided multiple invoices for comparison:

{comparison_text}

Your task:
1. Compare the provided invoices, highlighting differences in plans, add-ons, and total cost.
2. Identify the most cost-effective option based on the user's needs.
3. Suggest potential optimizations for each invoice.
4. Provide the comparison as **plain text** with embedded HTML-like formatting:
    - Use <b>...</b> to bold plan names, add-on names, and price differences.
    - Separate each invoice comparison with <hr>.
    - Summarize key differences in a <ul> list format.
"""

    prompt = f"""
The user asked: "{user_message}".

Previous conversation history:
{history_text}

User's current usage data:
{json.dumps(usage_details, indent=2)}

Available plans, add-ons, and pricing:
{json.dumps(services_data, indent=2)}

Your task:
1. Answer the user queries as accurately as possible based on the json data given
2. Analyze the user's usage data to determine the most cost-effective plan and add-ons.
3. Suggest the best invoice configuration, including the package name, add-ons, and total cost.
4. Provide a **detailed breakdown** of the recommended invoice:
    - Base plan details (name, price, features included).
    - Add-ons required to meet the usage needs.
    - Total cost calculation.
5. Format the response as **plain text** with embedded HTML-like formatting:
    - Use <b>...</b> for bolding key sections.
    - Separate sections with <hr>.
    - Use <p>...</p> for paragraphs and <ul>/<li> for lists.
6. If invoices are provided, compare and highlight key differences.
7. Ensure responses are detailed and specific to the user’s request.
8. Do not reccomend a plan if the user does not ask for it.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Help users find service plans."},
                {"role": "user", "content": prompt}
            ]
        )
        # Correct way to extract message content
        return (response.choices[0].message.content)
    except Exception as e:
        return f"OpenAI error: {e}"


def classify_query_intent(user_message):
    """
    Determines whether the user's query requires usage-based recommendations or informational response.
    """
    usage_keywords = ["usage", "optimize", "current plan", "my plan", "invoice", "recommend"]
    informational_keywords = ["what is", "services", "plans", "overview", "what are"]

    user_message_lower = user_message.lower()

    if any(keyword in user_message_lower for keyword in informational_keywords):
        return "informational"  # Provide only plan/service details
    
    if any(keyword in user_message_lower for keyword in usage_keywords):
        return "recommendation"  # Provide recommendation based on usage

    return "informational"  # Default to informational if unclear


# Save conversation to database


def save_conversation_to_db(session_id, user_message, bot_response):
    with db_pool.getconn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            INSERT INTO conversation (session_id, user_message, bot_response)
            VALUES (%s, %s, %s)
            """, (session_id, user_message, bot_response))
            conn.commit()

# Process user message


def process_user_message(user_message, session_id):
    """
    Processes the user's message to generate a chatbot response.
    """
    session = get_session(session_id)
    entities = extract_entities(user_message)

    # Determine if usage data should be used based on the query intent
    use_usage_data = classify_query_intent(user_message)
    # Provide usage details conditionally
    usage_details = services_data.get(
        "usage_details", None) if use_usage_data else None

    # Generate response based on user query and context
    response_message = call_gpt_function_suggestion(
        user_message,
        entities,
        session["history"],
        usage_details
    )

    # Save the current message and response to session history
    session["history"].append({"user": user_message, "bot": response_message})

    # Save the conversation to the database
    save_conversation_to_db(session_id, user_message, response_message)

    return response_message

# Routes


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message', '')
    session_id = request.json.get('session_id', str(uuid.uuid4()))
    response_message = process_user_message(user_message, session_id)
    return jsonify({"response": response_message, "session_id": session_id})


if __name__ == '__main__':
    app.run(debug=True)
