import os
import time
import base64
from io import BytesIO

import streamlit as st
import numpy as np
import pandas as pd
from openai import AzureOpenAI


# ==================== CONFIGURATION (Backend) ====================

# All sensitive values are taken from environment variables / Streamlit secrets
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")  # deployment name in Azure

if not AZURE_API_KEY or not AZURE_ENDPOINT:
    st.error("Azure OpenAI credentials are not set. Please configure AZURE_API_KEY and AZURE_ENDPOINT in Streamlit secrets.")
    st.stop()

# =================================================================
# Page configuration

st.set_page_config(
    page_title="Nutrition Advisor",
    page_icon="🥗",
    layout="wide",
)

# Optional custom CSS (currently empty – keep for future styling)
st.markdown(
    """
    <style>
    /* Add custom CSS here if needed */
    </style>
    """,
    unsafe_allow_html=True,
)

# App header
st.markdown("## 🥗 AI Nutrition Advisor")
st.markdown("Get personalized food recommendations & nutritional analysis")

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
            "Low Carb Diet",
        ],
    )

    st.divider()

    # Additional preferences
    st.header("🍽️ Preferences")
    meal_type = st.multiselect(
        "Meal Type (optional):",
        ["Breakfast", "Lunch", "Dinner", "Snack", "Pre-workout", "Post-workout"],
        default=[],
    )

    num_recommendations = st.slider(
        "Number of recommendations:",
        min_value=1,
        max_value=10,
        value=5,
        help="How many food suggestions would you like?",
    )

    dietary_restrictions = st.multiselect(
        "Dietary Restrictions (optional):",
        ["None", "Dairy-free", "Gluten-free", "Nut-free", "Vegetarian", "Vegan", "Halal", "Kosher"],
        default=["None"],
    )

    st.divider()
    st.caption("💡 Tip: Be specific in your questions for better recommendations!")

# Initialize session state
if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = []

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []


# ==================== Helper Functions ====================

def create_openai_client():
    """Create and return an AzureOpenAI client."""
    try:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
        )
        return client
    except Exception as e:
        st.error(f"Error creating OpenAI client: {str(e)}")
        return None


def encode_image(image_bytes: bytes) -> str:
    """Encode raw image bytes as base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def get_nutrition_recommendations(
    client,
    query: str,
    health_goal: str,
    num_recommendations: int,
    meal_type: list,
    dietary_restrictions: list,
) -> str | None:
    """Generate nutrition recommendations from text query and user settings."""
    prompt = f"""
You are a professional nutrition advisor. Based on the following information, provide {num_recommendations} specific food recommendations.

User's Question: {query}
Health Goal: {health_goal}
Meal Type: {', '.join(meal_type) if meal_type else 'Any meal'}
Dietary Restrictions: {', '.join(dietary_restrictions)}

Please provide exactly {num_recommendations} food recommendations.
For each recommendation, include:
1. Food/Meal name
2. Brief description (1-2 sentences)
3. Key nutritional benefits
4. Approximate calories (if relevant)
5. Why it fits the user's goal

Format your response in a clear, organized manner with numbered items.
""".strip()

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,  # deployment name, not raw model name
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a knowledgeable nutrition advisor who provides "
                        "evidence-based, practical food recommendations tailored "
                        "to individual health goals and dietary needs."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return None


def analyze_food_from_image(client, image_bytes: bytes, additional_query: str = "") -> str | None:
    """Analyze food from an uploaded image."""
    base64_image = encode_image(image_bytes)

    prompt = f"""
Analyze this food image and provide a detailed nutritional breakdown. Include:

1. **Food Identification**: What food items do you see?
2. **Estimated Portion Size**: Approximate serving size
3. **Nutritional Information**:
   - Calories (approximate)
   - Macronutrients (protein, carbs, fats in grams)
   - Key vitamins and minerals
4. **Health Assessment**: Is this meal healthy? Any concerns?
5. **Recommendations**: How could this meal be improved nutritionally?

{f'Additional Information: {additional_query}' if additional_query else ''}

Provide your analysis in a clear, structured format.
""".strip()

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None


def analyze_food_from_text(client, food_description: str) -> str | None:
    """Analyze food from a textual description."""
    prompt = f"""
Analyze the following food/meal description and provide a detailed nutritional breakdown:

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

Provide your analysis in a clear, structured format.
""".strip()

    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a nutrition expert who can analyze food descriptions "
                        "and provide detailed nutritional information and health recommendations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing food: {str(e)}")
        return None


# ==================== Main UI Layout ====================

tab1, tab2 = st.tabs(["🍴 Get Food Recommendations", "🔍 Analyze Nutritional Content"])

# -------------------- TAB 1: Food Recommendations --------------------
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask for Food Recommendations")
        user_query = st.text_area(
            "What kind of food recommendations are you looking for?",
            placeholder=(
                "E.g., 'What should I eat for breakfast to boost my energy?' "
                "or 'Suggest protein-rich snacks for muscle building'"
            ),
            height=100,
            key="recommendation_query",
        )

        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            submit_button = st.button(
                "🔍 Get Recommendations",
                type="primary",
                use_container_width=True,
            )
        with col_btn2:
            clear_rec_button = st.button(
                "🗑️ Clear History",
                use_container_width=True,
                key="clear_rec",
            )

        if clear_rec_button:
            st.session_state.recommendation_history = []
            st.rerun()

    with col2:
        st.header("📋 Current Settings")
        st.info(
            f"""
**Goal:** {health_goal}  
**Meal Types:** {', '.join(meal_type) if meal_type else 'Any'}  
**Recommendations:** {num_recommendations}  
**Restrictions:** {', '.join(dietary_restrictions)}
"""
        )

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
                        dietary_restrictions,
                    )

                if recommendations:
                    st.session_state.recommendation_history.append(
                        {
                            "query": user_query,
                            "goal": health_goal,
                            "response": recommendations,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    st.success("✅ Recommendations generated successfully!")

    # Display recommendation history
    if st.session_state.recommendation_history:
        st.header("📜 Recommendation History")
        for idx, chat in enumerate(reversed(st.session_state.recommendation_history)):
            with st.expander(
                f"🕒 {chat['timestamp']} - {chat['goal']}",
                expanded=(idx == 0),
            ):
                st.markdown(f"**Your Question:** {chat['query']}")
                st.divider()
                st.markdown("**AI Recommendations:**")
                st.markdown(chat["response"])


# -------------------- TAB 2: Nutritional Analysis --------------------
with tab2:
    st.header("🔍 Analyze Nutritional Content")

    analysis_mode = st.radio(
        "Choose input method:",
        ["Describe food in text", "Upload food image"],
        horizontal=True,
    )

    client = None

    if analysis_mode == "Describe food in text":
        food_description = st.text_area(
            "Describe your meal or food item:",
            placeholder="E.g., 'One bowl of beef pho with extra noodles and side of spring rolls'",
            height=120,
            key="analysis_text",
        )

        col_a1, col_a2 = st.columns([1, 5])
        with col_a1:
            analyze_text_btn = st.button(
                "📊 Analyze Text",
                type="primary",
                use_container_width=True,
            )
        with col_a2:
            clear_analysis_btn = st.button(
                "🧹 Clear History",
                use_container_width=True,
                key="clear_analysis",
            )

        if clear_analysis_btn:
            st.session_state.analysis_history = []
            st.rerun()

        if analyze_text_btn:
            if not food_description.strip():
                st.warning("⚠️ Please enter a food or meal description.")
            else:
                client = client or create_openai_client()
                if client:
                    with st.spinner("🔎 Analyzing nutritional content..."):
                        analysis_result = analyze_food_from_text(
                            client,
                            food_description.strip(),
                        )

                    if analysis_result:
                        st.session_state.analysis_history.append(
                            {
                                "mode": "text",
                                "input": food_description.strip(),
                                "result": analysis_result,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                        st.success("✅ Analysis completed!")

    else:  # Upload food image
        uploaded_image = st.file_uploader(
            "Upload a food image (JPG/PNG):",
            type=["jpg", "jpeg", "png"],
        )

        additional_query = st.text_input(
            "Optional: Add any extra information (e.g., 'I ate half of this plate', 'This was my lunch')",
            key="image_extra",
        )

        col_i1, col_i2 = st.columns([1, 5])
        with col_i1:
            analyze_image_btn = st.button(
                "🖼️ Analyze Image",
                type="primary",
                use_container_width=True,
            )
        with col_i2:
            clear_analysis_btn2 = st.button(
                "🧹 Clear History",
                use_container_width=True,
                key="clear_analysis2",
            )

        if clear_analysis_btn2:
            st.session_state.analysis_history = []
            st.rerun()

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded food image", use_column_width=True)

        if analyze_image_btn:
            if uploaded_image is None:
                st.warning("⚠️ Please upload a food image first.")
            else:
                client = client or create_openai_client()
                if client:
                    with st.spinner("🔎 Analyzing image and estimating nutrition..."):
                        image_bytes = uploaded_image.read()
                        analysis_result = analyze_food_from_image(
                            client,
                            image_bytes,
                            additional_query=additional_query.strip(),
                        )

                    if analysis_result:
                        st.session_state.analysis_history.append(
                            {
                                "mode": "image",
                                "input": uploaded_image.name,
                                "extra": additional_query.strip(),
                                "result": analysis_result,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                        st.success("✅ Image analysis completed!")

    # Display analysis history
    if st.session_state.analysis_history:
        st.header("📊 Analysis History")
        for idx, item in enumerate(reversed(st.session_state.analysis_history)):
            title_suffix = "Text" if item["mode"] == "text" else "Image"
            with st.expander(
                f"🕒 {item['timestamp']} - {title_suffix} analysis",
                expanded=(idx == 0),
            ):
                if item["mode"] == "text":
                    st.markdown(f"**Input Description:** {item['input']}")
                else:
                    st.markdown(f"**Image File:** {item['input']}")
                    if item.get("extra"):
                        st.markdown(f"**Extra Info:** {item['extra']}")

                st.divider()
                st.markdown("**AI Nutritional Analysis:**")
                st.markdown(item["result"])


# ==================== Footer / Disclaimer ====================

st.divider()
st.caption(
    """
⚠️ **Disclaimer:** This app provides AI-generated nutritional suggestions for informational purposes only.  
Always consult with a qualified healthcare professional or registered dietitian before making significant dietary changes.  

Powered by Azure OpenAI | Built with Streamlit
"""
)
