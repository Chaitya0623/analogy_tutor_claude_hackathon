"""
Analogy Tutor - Learn Any Topic Through Your Interests (Multi-Model)

Streamlit app that teaches technical concepts using analogies grounded in your
personal interests. Supports:
- Gemini 2.0 Flash
- OpenAI GPT-5.1-chat
- Claude 2.1

API keys should be set in a .env file:
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional
import os
import re
import random
from dotenv import load_dotenv

# Model clients
import google.generativeai as genai
import openai
import anthropic


# ============================================================================
# LOAD ENV
# ============================================================================

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_INTERESTS = [
    "How I Met Your Mother",
    "Taylor Swift",
    "cooking",
    "basketball"
]

AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gpt-5.1-chat",
    "claude-2.1"
]


# ============================================================================
# SMART INTEREST SELECTION
# ============================================================================

def pick_best_interest(topic: str, interests: list):
    topic_words = set(re.findall(r"\w+", topic.lower()))
    best_interest, best_score = None, -1
    explanations = []

    for interest in interests:
        interest_words = set(re.findall(r"\w+", interest.lower()))
        shared = topic_words.intersection(interest_words)
        score = len(shared)
        explanations.append(f"Interest '{interest}' matched {score} keywords: {shared}")
        if score > best_score:
            best_score = score
            best_interest = interest

    if best_score == 0:
        random_choice = random.choice(interests)
        reason = (
            "No keyword overlap found. Random interest chosen.\n\n"
            "Scoring summary:\n" + "\n".join(explanations)
        )
        return random_choice, reason

    reason = (
        f"Interest '{best_interest}' selected because it shared {best_score} keyword(s) with your topic.\n\n"
        "Scoring summary:\n" + "\n".join(explanations)
    )
    return best_interest, reason


# ============================================================================
# API SETUP
# ============================================================================

def setup_apis():
    keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "claude": os.getenv("ANTHROPIC_API_KEY")
    }

    # for name, key in keys.items():
    #     if not key:
    #         st.error(f"❌ Missing {name.upper()}_API_KEY in .env")

    if keys["gemini"]:
        genai.configure(api_key=keys["gemini"])

    return keys


API_KEYS = setup_apis()


# ============================================================================
# ANALOGY GENERATION
# ============================================================================

def generate_analogy(topic: str, interest: str, model_name: str) -> Optional[str]:
    prompt = f"""
You are an expert analogy-focused tutor. Generate a detailed, structured explanation of "{topic}" using "{interest}" as the analogy domain.

REQUIREMENTS:
1) State the interest domain and explain why it fits the topic.
2) Provide the analogy in exactly 3 short sentences.
3) Provide 5–8 bullet mappings from topic → analogy item.
4) Include one self-check question with an answer.
Follow the structure EXACTLY.
"""

    try:
        if model_name == "gemini-2.0-flash":
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text

        elif model_name == "gpt-5.1-chat":
            openai.api_key = API_KEYS["openai"]
            completion = openai.ChatCompletion.create(
                model="gpt-5.1-chat",
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content

        elif model_name == "claude-2.1":
            client = anthropic.Anthropic(api_key=API_KEYS["claude"])
            response = client.completions.create(
                model="claude-2.1",
                max_tokens_to_sample=800,
                prompt=prompt
            )
            return response["completion"]

    except Exception as e:
        st.error(f"Error generating analogy ({model_name}): {str(e)}")
        return None


# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    if "user_interests" not in st.session_state:
        st.session_state.user_interests = DEFAULT_INTERESTS.copy()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "chosen_model" not in st.session_state:
        st.session_state.chosen_model = AVAILABLE_MODELS[0]


def add_to_history(result: Dict):
    st.session_state.history.insert(0, result)
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[:10]


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    st.sidebar.title("⚙️ Settings")

    st.sidebar.subheader("🧠 Choose Model")
    st.session_state.chosen_model = st.sidebar.selectbox(
        "Model:", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(st.session_state.chosen_model)
    )

    st.sidebar.subheader("🎯 Your Interests")
    st.sidebar.caption("Analogies will be drawn from these domains.")
    for i, interest in enumerate(st.session_state.user_interests):
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(f"• {interest}")
        if col2.button("✕", key=f"remove_{i}"):
            st.session_state.user_interests.pop(i)
            st.rerun()

    with st.sidebar.form("add_interest_form"):
        new_interest = st.text_input("Add new interest:")
        if st.form_submit_button("➕ Add"):
            if new_interest and new_interest not in st.session_state.user_interests:
                st.session_state.user_interests.append(new_interest)
                st.rerun()

    if st.sidebar.button("🔄 Reset to Defaults"):
        st.session_state.user_interests = DEFAULT_INTERESTS.copy()
        st.rerun()


# ============================================================================
# RENDER ANALOGY
# ============================================================================

def render_analogy_result(result: Dict):
    st.success("✅ Analogy generated!")
    st.info(f"**🎬 Using interest domain:** {result['interest_domain']}")
    st.subheader(f"📚 Topic: {result['topic']}")
    st.divider()
    st.markdown(result["analogy_text"])
    st.divider()
    st.caption(f"💡 Generated using {result.get('source', 'Unknown')} ({st.session_state.chosen_model})")


# ============================================================================
# HISTORY
# ============================================================================

def render_history():
    if not st.session_state.history:
        return
    with st.expander("📖 Learning History", expanded=False):
        for i, item in enumerate(st.session_state.history):
            st.markdown(
                f"**{i+1}. Query:** {item['query']}  \n"
                f"**Interest Used:** {item['interest_domain']}  \n"
                f"**Source:** {item.get('source', 'Unknown')}"
            )
            if i < len(st.session_state.history) - 1:
                st.divider()


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(page_title="Analogy Tutor", page_icon="🎓", layout="wide")
    init_session_state()
    render_sidebar()

    st.title("🎓 Analogy Tutor")
    st.markdown("Learn ANY topic through analogies based on your interests!")

    st.subheader("🤔 What do you want to learn today?")
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input(
            "Enter a topic:",
            placeholder="e.g., backpropagation, gradient descent, transformers...",
            label_visibility="collapsed"
        )
    with col2:
        teach_button = st.button("🎯 Teach me with an analogy!", type="primary", use_container_width=True)

    if teach_button:
        if not user_query.strip():
            st.warning("⚠️ Please enter a topic!")
        else:
            chosen_interest, selection_reason = pick_best_interest(user_query, st.session_state.user_interests)
            with st.spinner(f"🔍 Generating analogy using {chosen_interest}..."):
                analogy_text = generate_analogy(user_query, chosen_interest, st.session_state.chosen_model)

            if analogy_text:
                result = {
                    "query": user_query,
                    "topic": user_query,
                    "interest_domain": chosen_interest,
                    "analogy_text": analogy_text,
                    "selection_reason": selection_reason,
                    "source": st.session_state.chosen_model
                }
                render_analogy_result(result)
                add_to_history(result)
            else:
                st.error("❌ Failed to generate analogy.")

    st.divider()
    render_history()
    st.markdown("---")
    st.caption("Powered by Gemini • OpenAI • Claude")


if __name__ == "__main__":
    main()