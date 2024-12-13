import openai
from flask import Flask, request, jsonify, send_from_directory
import json
import re
import spacy
import uuid
import psycopg2
from psycopg2 import sql, extras
from psycopg2.pool import SimpleConnectionPool
import os
from werkzeug.utils import secure_filename

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
openai.api_key = ''

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

# Analyze optimal plan with detailed feature usage comparison


def analyze_optimal_plan(current_plan, current_usage, plans):
    """
    Analyze the user's current plan and suggest the best alternative plan
    based on their usage.
    """
    current_plan_name = current_plan.get("name", "Unknown")
    current_plan_features = current_plan.get("features", {})
    current_plan_price = current_plan.get("price", 0)

    # Prepare usage analysis report
    usage_summary = []
    for feature, usage in current_usage.items():
        max_limit = current_plan_features.get(feature, 0)
        usage_summary.append({
            "feature": feature,
            "used": usage,
            "allowed": max_limit
        })

    # Suggest the best plan based on usage
    recommendations = []
    for plan_name, plan_details in plans.items():
        plan_cost = plan_details['price']
        plan_features = plan_details.get('features', {})
        overused = []
        underused = []
        total_cost = plan_cost

        for feature, usage in current_usage.items():
            limit = plan_features.get(feature, 0)
            if usage > limit:
                overused.append(
                    {"feature": feature, "usage": usage, "limit": limit})
            elif usage < 0.5 * limit:
                underused.append(
                    {"feature": feature, "usage": usage, "limit": limit})

        recommendations.append({
            "plan": plan_name,
            "cost": total_cost,
            "overused_features": overused,
            "underused_features": underused
        })

    # Sort recommendations by cost and suitability
    recommendations.sort(key=lambda x: (
        len(x['overused_features']), x['cost']))
    best_plan = recommendations[0]

    # Add recommendations to the report
    return {
        "current_plan": {
            "name": current_plan_name,
            "price": current_plan_price
        },
        "usage_summary": usage_summary,
        "recommendation": best_plan
    }


def calculate_add_on_cost(add_ons, feature, units_needed):
    """
    Calculate the cost of add-ons for a feature based on required units.

    Args:
        add_ons (dict): Available add-ons and their pricing.
        feature (str): Feature name (e.g., "Basic monitors").
        units_needed (int): Number of additional units required.

    Returns:
        float: Total cost for the required add-ons.
        list: List of add-ons used to meet the requirement.
    """
    feature_add_ons = add_ons.get(feature, {})
    unit_prices = [(int(units.split()[0]), price)
                   for units, price in feature_add_ons.items()]
    unit_prices.sort()  # Sort by units to find the best match first

    total_cost = 0
    remaining_units = units_needed
    selected_add_ons = []

    for units, price in unit_prices:
        if remaining_units <= 0:
            break
        # Calculate how many batches of this add-on are required
        batches_needed = (remaining_units + units - 1) // units  # Round up
        batch_units = batches_needed * units
        total_cost += batches_needed * price
        remaining_units -= batch_units
        selected_add_ons.append(
            {"units": batch_units, "price": batches_needed * price})

    if remaining_units > 0:
        raise ValueError(
            f"Insufficient add-on options to cover {remaining_units} units for feature '{feature}'.")

    return total_cost, selected_add_ons

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
        comparison_text = "\n\n".join(
            [f"Invoice {i + 1}:\n{json.dumps(invoice, indent=2)}" for i, invoice in enumerate(comparison_invoices)])
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
4. if the cost of the plan is higher than the usage you can downgrade the plan based on their usage
5. Provide a **detailed breakdown** of the recommended invoice:
    - Base plan details (name, price, features included).
    - Add-ons required to meet the usage needs.
    - Total cost calculation.
6. Format the response as **plain text** with embedded HTML-like formatting:
    - Use <b>...</b> for bolding key sections.
    - Separate sections with <hr>.
    - Use <p>...</p> for paragraphs and <ul>/<li> for lists.
7. If invoices are provided, compare and highlight key differences.
8. Ensure responses are detailed and specific to the user’s request.
9. Do not reccomend a plan if the user does not ask for it.
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
    usage_keywords = ["usage", "optimize",
                      "current plan", "my plan", "invoice", "recommend"]
    informational_keywords = ["what is",
                              "services", "plans", "overview", "what are"]

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


def generate_recommendation_html(current_plan, usage_summary, recommended_plan, addons, total_cost_details):
    """
    Dynamically generates HTML for the cost recommendation based on the provided data.

    Args:
        current_plan (dict): Details of the current plan.
        usage_summary (list of dict): List of current usage details.
        recommended_plan (dict): Details of the recommended plan.
        addons (list of dict): List of add-ons.
        total_cost_details (dict): Total cost breakdown.

    Returns:
        str: HTML content for the recommendation.
    """
    # Current Plan Section
    current_plan_html = f"""
    <h3>Current Plan</h3>
    <p><b>Plan:</b> {current_plan.get('name')}</p>
    <p><b>Next Payment Date:</b> {current_plan.get('next_payment_date')}</p>
    <h4>Usage Summary</h4>
    <ul>
    """
    for item in usage_summary:
        current_plan_html += f"<li><b>{item['feature']}:</b> {item['used']} used out of {item['allowed']} allowed</li>"
    current_plan_html += "</ul>"

    # Recommended Plan Section
    recommended_plan_html = f"""
    <h3>Recommended Plan</h3>
    <p><b>Plan:</b> {recommended_plan.get('name')}</p>
    <p><b>Price:</b> INR {recommended_plan.get('price')}</p>
    <h4>Included Features</h4>
    <ul>
    """
    for feature in recommended_plan.get('features', []):
        recommended_plan_html += f"<li>{feature}</li>"
    recommended_plan_html += "</ul>"

    # Add-ons Section
    addons_html = "<h3>Add-ons for Consideration</h3><ul>"
    for addon in addons:
        addons_html += f"""
        <li><b>{addon['name']}:</b>
            <ul>
                <li>{addon['details']} - INR {addon['price']}</li>
            </ul>
        </li>
        """
    addons_html += "</ul>"

    # Total Cost Section
    total_cost_html = "<h3>Total Cost Calculation</h3><ul>"
    for item in total_cost_details.get('breakdown', []):
        total_cost_html += f"<li>{item['name']}: INR {item['price']}</li>"
    total_cost_html += f"</ul><p><b>Total Cost:</b> INR {total_cost_details.get('total')}</p>"

    # Combine all sections
    html_output = f"""
    <p><b>Based on your current usage and requirements, here are my recommendations:</b></p>
    <hr>
    {current_plan_html}
    <hr>
    {recommended_plan_html}
    <hr>
    {addons_html}
    <hr>
    {total_cost_html}
    <p>This configuration allows you to manage current usage with room for growth while keeping costs optimized.</p>
    """
    return html_output


# Process user message
def process_user_message(user_message, session_id):
    session = get_session(session_id)
    entities = extract_entities(user_message)

    # Retrieve usage data if available
    usage_details = session.get("usage_data", None)

    # Classify query intent
    query_intent = classify_query_intent(user_message)

    if query_intent == "recommendation" and not usage_details:
        return "Please upload your usage data before requesting recommendations."

    # Generate response
    response_message = call_gpt_function_suggestion(
        user_message,
        entities,
        session["history"],
        usage_details
    )

    # Save conversation to session and database
    session["history"].append({"user": user_message, "bot": response_message})
    save_conversation_to_db(session_id, user_message, response_message)

    return response_message

# New endpoint to upload usage JSON


@app.route('/upload_usage', methods=['POST'])
def upload_usage():
    session_id = request.json.get('session_id', str(uuid.uuid4()))
    usage_data = request.json.get('usage_data')

    if not usage_data:
        return jsonify({"error": "No usage data provided"}), 400

    try:
        # Validate usage data
        if not isinstance(usage_data, dict):
            raise ValueError("Invalid usage data format.")

        session = get_session(session_id)
        session["usage_data"] = usage_data  # Save to session
        return jsonify({"message": "Usage data uploaded successfully", "session_id": session_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Routes


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/analyze_plan', methods=['POST'])
def analyze_plan():
    file = request.files.get('usage_file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        usage_data = json.load(file)

        # Extract the relevant plans and add-ons from all services
        all_plans = {}
        all_add_ons = {}

        for service_name, service_details in services_data["services"].items():
            if "plans" in service_details:
                all_plans.update(service_details["plans"])
            if "add_ons" in service_details:
                all_add_ons.update(service_details["add_ons"])

        if not all_plans or not all_add_ons:
            return jsonify({"error": "No valid plans or add-ons found in services data."}), 400

        # Analyze the plan and get recommendations
        optimal_plan = analyze_optimal_plan(
            current_usage=usage_data,
            plans=all_plans,
            add_ons=all_add_ons
        )

        # Prepare table data for frontend
        table_data = []
        for feature, usage in usage_data.items():
            current_plan = optimal_plan.get("plan", "N/A")
            suggested_plan = optimal_plan["plan"]
            savings = optimal_plan.get("total_cost", "N/A")
            table_data.append({
                "feature": feature,
                "current_plan": current_plan,
                "usage": usage,
                "suggested_plan": suggested_plan,
                "savings": savings
            })

        return jsonify({"table_data": table_data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/cost_analyzer')
def cost_analyzer():
    return send_from_directory('static', 'cost_analyzer.html')


@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message', '')
    session_id = request.json.get('session_id', str(uuid.uuid4()))
    # Retrieve usage data if provided
    usage_data = request.json.get('usage_data', None)

    # Store usage data in session if available
    session = get_session(session_id)
    if usage_data:
        session["usage_data"] = usage_data

    response_message = process_user_message(user_message, session_id)
    return jsonify({"response": response_message, "session_id": session_id})


if __name__ == '__main__':
    app.run(debug=True)
