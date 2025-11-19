import streamlit as st
import numpy as np
import pandas as pd
from openai import AzureOpenAI
import time
import base64
from io import BytesIO

# ==================== CONFIGURATION (Backend) ====================
# Insert your API credentials here
AZURE_API_KEY = "a0bccf54e1e14deab0feb9ea8217fd5d"  # Replace with your actual API key
AZURE_API_VERSION = "2023-05-15"
AZURE_ENDPOINT = "https://hkust.azure-api.net"
# =================================================================

# Page configuration
st.set_page_config(
    page_title="Nutrition Advisor",
    page_icon="🥗",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-box {
        background-color: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E7D32;
        margin: 1rem 0;
    }
    .nutrition-analysis-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1976d2;
        margin: 1rem 0;
        color: #000;
    }
    .dark-mode .recommendation-box {
        background-color: #1e3a1e;
    }
    .dark-mode .nutrition-analysis-box {
        background-color: #1e3a3a;
        color: #fff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">🥗 AI Nutrition Advisor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Get personalized food recommendations & nutritional analysis</p>', unsafe_allow_html=True)

# Sidebar for preferences
with st.sidebar:
    # Health goals selection
    st.header("🎯 Your Health Goal")
    health_goal = st.selectbox(
        "Select your primary goal:",
        [
            "General Healthy Eating",
            "Weight Loss",
            "Muscle Building",
            "Keep Fit/Maintenance",
            "Heart Health",
            "Energy Boost",
            "Diabetes Management",
            "High Protein Diet",
            "Vegetarian/Vegan",
            "Low Carb Diet"
        ]
    )

    st.divider()

    # Additional preferences
    st.header("🍽️ Preferences")

    meal_type = st.multiselect(
        "Meal Type (optional):",
        ["Breakfast", "Lunch", "Dinner", "Snack", "Pre-workout", "Post-workout"],
        default=[]
    )

    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many food suggestions would you like?"
    )

    dietary_restrictions = st.multiselect(
        "Dietary Restrictions (optional):",
        ["None", "Dairy-free", "Gluten-free", "Nut-free", "Vegetarian", "Vegan", "Halal", "Kosher"],
        default=["None"]
    )

    st.divider()
    st.caption("💡 Tip: Be specific in your questions for better recommendations!")

# Initialize session state
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Function to create OpenAI client
def create_openai_client():
    try:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT
        )
        return client
    except Exception as e:
        st.error(f"Error creating OpenAI client: {str(e)}")
        return None

# Function to encode image to base64
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Function to generate nutrition recommendations
def get_nutrition_recommendations(client, query, health_goal, num_recommendations, meal_type, dietary_restrictions):
    prompt = f"""You are a professional nutrition advisor. Based on the following information, provide {num_recommendations} specific food recommendations.

User's Question: {query}

Health Goal: {health_goal}
Meal Type: {', '.join(meal_type) if meal_type else 'Any meal'}
Dietary Restrictions: {', '.join(dietary_restrictions)}

Please provide exactly {num_recommendations} food recommendations. For each recommendation, include:
1. Food/Meal name
2. Brief description (1-2 sentences)
3. Key nutritional benefits
4. Approximate calories (if relevant)
5. Why it fits the user's goal

Format your response in a clear, organized manner with numbered items."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a knowledgeable nutrition advisor who provides evidence-based, practical food recommendations tailored to individual health goals and dietary needs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None

# Function to analyze food from image
def analyze_food_from_image(client, image_bytes, additional_query=""):
    base64_image = encode_image(image_bytes)

    prompt = f"""Analyze this food image and provide a detailed nutritional breakdown. Include:

1. **Food Identification**: What food items do you see?
2. **Estimated Portion Size**: Approximate serving size
3. **Nutritional Information**:
   - Calories (approximate)
   - Macronutrients (protein, carbs, fats in grams)
   - Key vitamins and minerals
4. **Health Assessment**: Is this meal healthy? Any concerns?
5. **Recommendations**: How could this meal be improved nutritionally?

{f'Additional Information: {additional_query}' if additional_query else ''}

Provide your analysis in a clear, structured format."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None

# Function to analyze food from text description
def analyze_food_from_text(client, food_description):
    prompt = f"""Analyze the following food/meal description and provide a detailed nutritional breakdown:

Food Description: {food_description}

Please provide:
1. **Food/Meal Summary**: Brief overview of what was described
2. **Estimated Nutritional Information**:
   - Calories (approximate)
   - Macronutrients (protein, carbs, fats in grams)
   - Key vitamins and minerals
3. **Health Assessment**: Is this meal healthy? Any nutritional concerns?
4. **Recommendations**: How could this meal be improved nutritionally?
5. **Suitable For**: What health goals does this meal support?

Provide your analysis in a clear, structured format."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a nutrition expert who can analyze food descriptions and provide detailed nutritional information and health recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing food: {str(e)}")
        return None

# Main tabs
tab1, tab2 = st.tabs(["🍴 Get Food Recommendations", "🔍 Analyze Nutritional Content"])

# ===================== TAB 1: Food Recommendations =====================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask for Food Recommendations")

        user_query = st.text_area(
            "What kind of food recommendations are you looking for?",
            placeholder="E.g., 'What should I eat for breakfast to boost my energy?' or 'Suggest protein-rich snacks for muscle building'",
            height=100,
            key="recommendation_query"
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            submit_button = st.button("🔍 Get Recommendations", type="primary", use_container_width=True)
        with col_btn2:
            clear_rec_button = st.button("🗑️ Clear History", use_container_width=True, key="clear_rec")

        if clear_rec_button:
            st.session_state.recommendation_history = []
            st.rerun()

    with col2:
        st.header("📋 Current Settings")
        st.info(f"""
        **Goal:** {health_goal}

        **Meal Types:** {', '.join(meal_type) if meal_type else 'Any'}

        **Recommendations:** {num_recommendations}

        **Restrictions:** {', '.join(dietary_restrictions)}
        """)

    # Handle recommendation submission
    if submit_button:
        if not user_query:
            st.warning("⚠️ Please enter your question or food preference.")
        else:
            client = create_openai_client()
            if client:
                with st.spinner("🤔 Generating personalized recommendations..."):
                    recommendations = get_nutrition_recommendations(
                        client,
                        user_query,
                        health_goal,
                        num_recommendations,
                        meal_type,
                        dietary_restrictions
                    )

                    if recommendations:
                        st.session_state.recommendation_history.append({
                            'query': user_query,
                            'goal': health_goal,
                            'response': recommendations,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.success("✅ Recommendations generated successfully!")

    # Display recommendation history
    if st.session_state.recommendation_history:
        st.header("📜 Recommendation History")
        for idx, chat in enumerate(reversed(st.session_state.recommendation_history)):
            with st.expander(f"🕒 {chat['timestamp']} - {chat['goal']}", expanded=(idx==0)):
                st.markdown(f"**Your Question:** {chat['query']}")
                st.divider()
                st.markdown(f"**AI Recommendations:**")
                st.markdown(f'<div class="recommendation-box">{chat["response"]}</div>', unsafe_allow_html=True)

# ===================== TAB 2: Nutritional Analysis =====================
with tab2:
    st.header("🔍 Analyze Nutritional Content")

    analysis_method = st.radio(
        "Choose analysis method:",
        ["📸 Upload Food Photo", "📝 Describe Food in Text"],
        horizontal=True
    )

    if analysis_method == "📸 Upload Food Photo":
        st.subheader("Upload a photo of your food")

        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of your food for nutritional analysis"
        )

        additional_context = st.text_input(
            "Additional Information (optional):",
            placeholder="E.g., 'grilled chicken breast, 200g' or 'homemade pasta with tomato sauce'",
            key="image_context"
        )

        if uploaded_file is not None:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(uploaded_file, caption="Uploaded Food Image", use_container_width=True)

            with col2:
                if st.button("🔬 Analyze Food", type="primary", use_container_width=True, key="analyze_image"):
                    client = create_openai_client()
                    if client:
                        with st.spinner("🧠 Analyzing nutritional content..."):
                            image_bytes = uploaded_file.getvalue()
                            analysis = analyze_food_from_image(client, image_bytes, additional_context)

                            if analysis:
                                st.session_state.analysis_history.append({
                                    'method': 'image',
                                    'context': additional_context if additional_context else 'No additional context',
                                    'analysis': analysis,
                                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                                })
                                st.success("✅ Analysis complete!")

    else:  # Text description
        st.subheader("Describe the food you want to analyze")

        food_description = st.text_area(
            "Describe your food/meal:",
            placeholder="E.g., 'One bowl of brown rice with grilled salmon, steamed broccoli, and avocado' or 'Two slices of whole wheat toast with peanut butter and banana'",
            height=100,
            key="food_description"
        )

        if st.button("🔬 Analyze Food", type="primary", key="analyze_text"):
            if not food_description:
                st.warning("⚠️ Please describe the food you want to analyze.")
            else:
                client = create_openai_client()
                if client:
                    with st.spinner("🧠 Analyzing nutritional content..."):
                        analysis = analyze_food_from_text(client, food_description)

                        if analysis:
                            st.session_state.analysis_history.append({
                                'method': 'text',
                                'description': food_description,
                                'analysis': analysis,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            st.success("✅ Analysis complete!")

    # Clear analysis history button
    if st.session_state.analysis_history:
        if st.button("🗑️ Clear Analysis History", key="clear_analysis"):
            st.session_state.analysis_history = []
            st.rerun()

    # Display analysis history
    if st.session_state.analysis_history:
        st.header("📊 Analysis History")
        for idx, analysis_item in enumerate(reversed(st.session_state.analysis_history)):
            with st.expander(f"🕒 {analysis_item['timestamp']} - {analysis_item['method'].upper()} Analysis", expanded=(idx==0)):
                if analysis_item['method'] == 'image':
                    st.markdown(f"**Additional Information:** {analysis_item['context']}")
                else:
                    st.markdown(f"**Food Description:** {analysis_item['description']}")
                st.divider()
                st.markdown(f"**Nutritional Analysis:**")
                st.markdown(f'<div class="nutrition-analysis-box">{analysis_item["analysis"]}</div>', unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>⚠️ <strong>Disclaimer:</strong> This app provides AI-generated nutritional suggestions for informational purposes only.
    Always consult with a qualified healthcare professional or registered dietitian before making significant dietary changes.</p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem;'>Powered by Azure OpenAI GPT-4o | Built with Streamlit</p>
</div>

""", unsafe_allow_html=True)
