"""Customer Service Assistant - Sequential Query Processing

This implements the correct flow:
1. Show one query at a time (sequential)
2. DS agent processes query
3. If clarification needed → human responds
4. When resolved → next query
5. Repeat for all 91 queries
"""

import streamlit as st
import sys
from pathlib import Path
import os
import json
import pandas as pd
import datetime
import uuid
import html
import re
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
from typing import Dict, List, Tuple, Optional
import plotly.express as px
from dotenv import load_dotenv
from urllib.parse import urlparse, urlunparse, parse_qs, parse_qsl, urlencode, unquote

# Add root path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env (for local development)
load_dotenv()

from src.models.ds_mass_function import DSMassFunction
from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
from src.models.classifier import IntentClassifier
from src.utils.explainability import BeliefTracker, BeliefVisualizer
from src.utils.data_logger import init_logger, save_session_to_github
from config.hierarchy_loader import (
    load_hierarchy_from_json,
    load_hierarchical_intents_from_json
)
from config.threshold_loader import load_thresholds_from_json


# ===== QUALTRICS/PROLIFIC INTEGRATION =====
def _get_query_params():
    """Get URL query parameters from Streamlit"""
    return st.query_params

def _as_str(val):
    """Convert parameter value to string safely"""
    if isinstance(val, list):
        return val[0] if val else ""
    return str(val) if val else ""

def _is_safe_return(ru):
    """Validate that return URL points to Qualtrics domain"""
    if not ru:
        return False
    try:
        d = unquote(ru)
        if not d.startswith(("http://", "https://")):
            d = "https://" + d
        p = urlparse(d)
        return (p.scheme in ("http", "https")) and ("qualtrics.com" in p.netloc)
    except Exception:
        return False

def _build_final_return(done=True):
    """Build Qualtrics return URL with participant data appended"""
    rr = st.session_state.get("return_raw", "")
    if not rr or not _is_safe_return(rr):
        return None
    decoded = unquote(rr)
    if not decoded.startswith(("http://", "https://")):
        decoded = "https://" + decoded
    p = urlparse(decoded)
    q = dict(parse_qsl(p.query, keep_blank_values=True))  # Use parse_qsl like anthrokit
    
    # Append parameters if missing (for data linkage in Qualtrics)
    prolific_pid_ss = st.session_state.get("prolific_pid", "")
    session_id_ss = st.session_state.get("session_id", "")  # KEY for data linkage
    
    if "PROLIFIC_PID" not in q and prolific_pid_ss:
        q["PROLIFIC_PID"] = prolific_pid_ss
    if "pid" not in q and st.session_state.get("pid"):
        q["pid"] = st.session_state.pid
    if "cond" not in q and st.session_state.get("cond"):
        q["cond"] = st.session_state.cond
    if "session_id" not in q and session_id_ss:
        q["session_id"] = session_id_ss  # Pass session_id back for data linkage
    if "done" not in q:
        q["done"] = "1" if done else "0"
    
    return urlunparse(p._replace(query=urlencode(q, doseq=True)))

def back_to_survey(done_flag=True):
    """Redirect back to Qualtrics/Prolific survey after study completion"""
    if st.session_state.get("_returned", False):
        return
    final = _build_final_return(done=done_flag)
    if not final:
        st.warning("WARNING: Return link missing or invalid. Please close this window and return to the survey manually.")
        return
    st.session_state._returned = True
    # Immediate redirect using meta refresh
    st.markdown(f'<meta http-equiv="refresh" content="0;url={final}">',
                unsafe_allow_html=True)
    st.info("Redirecting you back to the survey...")
    st.stop()

# Persist URL parameters once at session start
_qs = _get_query_params()
_pid_in = _as_str(_qs.get("pid", ""))
_cond_in = _as_str(_qs.get("cond", ""))
_ret_in = _as_str(_qs.get("return", ""))
_prolific_pid = _as_str(_qs.get("PROLIFIC_PID", ""))

if "pid" not in st.session_state and _pid_in:
    st.session_state.pid = _pid_in
if "cond" not in st.session_state and _cond_in:
    st.session_state.cond = _cond_in
if "return_raw" not in st.session_state and _ret_in:
    st.session_state.return_raw = _ret_in
if "prolific_pid" not in st.session_state and _prolific_pid:
    st.session_state.prolific_pid = _prolific_pid

# One-shot redirect latch
if "_returned" not in st.session_state:
    st.session_state._returned = False

# Store back_to_survey function in session state for access from UI
st.session_state.back_to_survey = back_to_survey
# ===== END QUALTRICS/PROLIFIC INTEGRATION =====


def _get_openai_client():
    """Get an OpenAI client if API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = None
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL")
    try:
        from openai import OpenAI  # type: ignore
        if base_url:
            return OpenAI(api_key=api_key, base_url=base_url)
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _llm_configured() -> bool:
    """Return True when an OpenAI API key is available."""
    if os.getenv("OPENAI_API_KEY"):
        return True
    try:
        return bool(st.secrets.get("OPENAI_API_KEY", None))
    except Exception:
        return False


def _test_llm_connection() -> Tuple[bool, str]:
    """Run a minimal LLM request to validate configuration."""
    client = _get_openai_client()
    if client is None:
        return False, "OpenAI client not available. Check OPENAI_API_KEY."

    model_name = st.session_state.get(
        "llm_model_name",
        os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a test endpoint. Reply with 'ok'."},
                {"role": "user", "content": "ping"}
            ],
            temperature=0.0,
            max_tokens=5
        )
        content = completion.choices[0].message.content if completion and completion.choices else ""
        if content and "ok" in content.lower():
            return True, f"LLM test passed using model '{model_name}'."
        return True, f"LLM responded, but output was unexpected: '{content}'."
    except Exception as e:
        return False, f"LLM test failed: {str(e)}"


def _extract_options_from_clarification(text: str) -> str:
    """Extract option list from a clarification prompt, if present."""
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return ""


def _humanize_response(text: str, response_type: str, context: Optional[Dict[str, str]] = None) -> str:
    """Optionally rewrite responses with LLM for a more natural tone."""
    if not text:
        return text
    if not st.session_state.get("humanize_responses", False):
        return text

    client = _get_openai_client()
    if client is None:
        return text

    model_name = st.session_state.get(
        "llm_model_name",
        os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    
    # Use higher temperature for explanations to get more natural variation
    if response_type == "explanation":
        default_temp = "0.8"
    else:
        default_temp = "0.6"
    
    temperature = float(
        st.session_state.get(
            "llm_temperature",
            os.getenv("TEMPERATURE", default_temp)
        )
    )

    if response_type == "clarification":
        system_prompt = (
            "You are a friendly customer service assistant having a natural conversation. "
            "Rewrite the clarification question to sound warm, helpful, and conversational. "
            "Use contractions (I'm, you're, etc.) and casual language. "
            "Ask questions like a helpful human would, not like a form or menu. "
            "Keep it brief and natural."
        )
    elif response_type == "prediction":
        system_prompt = (
            "You are a friendly customer service assistant confirming you understood the customer. "
            "Rewrite to sound warm, natural, and reassuring - like you're genuinely helping. "
            "Use contractions (I'm, you're, etc.) and keep it conversational. "
            "Make the customer feel heard and understood. "
            "Keep it short, friendly, and to the point."
        )
    elif response_type == "explanation":
        system_prompt = (
            "You are a friendly customer service assistant explaining how you figured out what the customer needed. "
            "Rewrite the explanation to sound like you're genuinely sharing your thought process with a friend. "
            "Use natural, conversational language with contractions (I'm, you're, wasn't, etc.). "
            "Make it feel like casual conversation, not a technical explanation. "
            "Vary your phrasing - don't use the exact same words every time. "
            "Keep it brief, warm, and human. Sound like you're really talking to someone."
        )
    else:
        system_prompt = (
            "You are a friendly customer service assistant having a natural conversation. "
            "Rewrite this message to sound warm, helpful, and conversational. "
            "Use contractions and casual language. "
            "Preserve all factual information but make it friendly and accessible."
        )

    ctx_lines = []
    if context:
        for key, value in context.items():
            if value:
                ctx_lines.append(f"- {key}: {value}")
    ctx_blob = "\n".join(ctx_lines) if ctx_lines else "(none)"

    user_prompt = (
        "Rewrite this message to sound like natural conversation with a customer. "
        "Be warm, friendly, and helpful. Use contractions and casual language. "
        "Vary your phrasing - don't repeat the same words or sentence structure. "
        "Make it feel like you're genuinely talking to someone, not reading from a script. "
        "Sound spontaneous and human, like real conversation.\n\n"
        f"Context:\n{ctx_blob}\n\n"
        f"Message to rewrite:\n{text}\n\n"
        "Your natural, conversational version:"
    )

    try:
        max_tokens = int(
            st.session_state.get(
                "llm_max_tokens",
                int(os.getenv("MAX_TOKENS", "400"))
            )
        )
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        content = completion.choices[0].message.content if completion and completion.choices else None
        return content or text
    except Exception:
        return text

# Page config
st.set_page_config(
    page_title="Customer Service Assistant", 
    page_icon="B",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS styling
st.markdown("""
<style>
.main { padding: 2rem 1rem; }
.header-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px; border-radius: 10px; margin-bottom: 30px;
    color: white; text-align: center;
}
.query-card {
    background: white; border: 1px solid #e1e8ed;
    border-radius: 10px; padding: 15px; margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.user-message { 
    background: #007bff; color: white; padding: 12px 16px;
    border-radius: 18px 18px 4px 18px; margin: 10px 0; text-align: right;
}
.bot-message {
    background: white; border: 1px solid #e1e8ed; padding: 12px 16px;
    border-radius: 18px 18px 18px 4px; margin: 10px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_ds_system():
    """Initialize DS system components - cached for speed like LLM simulation"""
    try:
        # Load hierarchy and intents - same paths as LLM simulation
        hierarchy_path = 'config/hierarchies/banking77_hierarchy.json'
        intents_path = 'config/hierarchies/banking77_intents.json' 
        threshold_path = 'results/banking77/workflow_demo/banking77_optimal_thresholds.json'  # Use computed thresholds
        
        hierarchy = load_hierarchy_from_json(hierarchy_path)
        hierarchical_intents = load_hierarchical_intents_from_json(intents_path)
        
        # Initialize embedder and intent embeddings - same as LLM simulation
        embedder = SentenceEmbedder(model_name='intfloat/e5-base')
        intent_embeddings = IntentEmbeddings(hierarchical_intents, embedder=embedder)
        
        # Load trained classifier - download from Dropbox if needed
        classifier_path = 'experiments/banking77/banking77_logistic_model.pkl'
        
        # Download model from Dropbox if not available locally
        if not os.path.exists(classifier_path):
            st.info("Downloading model from Dropbox...")
            try:
                from src.utils.dropbox_saver import download_model_from_dropbox
                downloaded_path = download_model_from_dropbox(
                    model_name='banking77_logistic_model.pkl',
                    local_dir='experiments/banking77'
                )
                if downloaded_path:
                    st.success("Model downloaded successfully!")
                    classifier_path = downloaded_path
                else:
                    st.error("Failed to download model from Dropbox")
                    st.stop()
            except Exception as e:
                st.error(f"Dropbox download failed: {e}")
                st.stop()
        
        classifier = IntentClassifier.from_pretrained(classifier_path)
        
        # Load optimal thresholds computed in STEP 2
        custom_thresholds = None
        if os.path.exists(threshold_path):
            custom_thresholds = load_thresholds_from_json(threshold_path)
        
        # Initialize DS system - Enable belief tracking for explainability
        ds_system = DSMassFunction(
            intent_embeddings=intent_embeddings.get_all_embeddings(),
            hierarchy=hierarchy,
            classifier=classifier,
            custom_thresholds=custom_thresholds,
            enable_belief_tracking=True,  # Enable for real explainability
            intent_descriptions=hierarchical_intents  # Pass descriptions for conversational clarifications
        )
        
        return ds_system
        
    except Exception as e:
        st.error(f"System initialization failed: {str(e)}")
        st.stop()

@st.cache_data
def load_study_queries():
    """Load the 91 study queries"""
    try:
        # Try multiple possible paths
        possible_paths = [
            'sample_study_queries.csv',
            'outputs/user_study/workflow_demo/selected_queries_for_user_study.csv',
            'data/user_study_queries.csv'
        ]
        
        queries_path = None
        for path in possible_paths:
            if os.path.exists(path):
                queries_path = path
                break
        
        if queries_path is None:
            st.error("Study queries file not found!")
            st.stop()
        
        queries_df = pd.read_csv(queries_path)
        return queries_df
        
    except Exception as e:
        st.error(f"Failed to load study queries: {str(e)}")
        st.stop()

def show_header():
    """Display header with HiCXAI styling"""
    st.markdown("""
    <style>
    .main { padding: 2rem 1rem; }
    .stApp { background-color: #fafafa; }

    .query-card {
        background: white; border: 1px solid #e1e8ed;
        border-radius: 10px; padding: 15px; margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .query-card:hover {
        border-color: #007bff; box-shadow: 0 4px 8px rgba(0,123,255,0.2);
    }
    .interaction-container { max-width: 100%; margin: 0 auto; }
    .user-message {
        background: #007bff; color: white; padding: 12px 16px;
        border-radius: 18px 18px 4px 18px; margin: 10px 0; text-align: right;
    }
    .bot-message {
        background: white; border: 1px solid #e1e8ed; padding: 12px 16px;
        border-radius: 18px 18px 18px 4px; margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 35px 30px; border-radius: 12px; margin-bottom: 35px;
        color: white; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-container">
        <h2 style="margin: 0; font-size: 2em;">Customer Service Assistant</h2>
        <p style="margin: 15px 0 8px 0; opacity: 0.95; line-height: 1.5; font-size: 1.1em;">
            This customer service assistant responds to customer support queries. I will process each initial message (Customer Query) and ask clarifying questions if needed to improve my understanding.
        </p>
        <p style="margin: 8px 0 0 0; opacity: 0.95; line-height: 1.5; font-size: 1.1em;">
            Please respond as a real user would — naturally and honestly, without trying to help or mislead the system.
        </p>
        <p style="margin: 8px 0 0 0; opacity: 0.9; font-size: 1em; line-height: 1.4;">
            <strong>For each query:</strong>
        </p>
        <ul style="margin: 5px 0 0 20px; opacity: 0.9; font-size: 0.95em; line-height: 1.6; text-align: left; display: inline-block;">
            <li>I may ask clarifying questions to better understand your needs</li>
            <li>Once resolved, you'll validate which option best matches your intent</li>
            <li>Then we'll proceed to the next query</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def format_bubble_text(text: str) -> str:
    """Format text for HTML chat bubbles with basic markdown support."""
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = safe.replace("\n", "<br>")
    return safe

def process_query(query_text, ds_system, is_initial=True, previous_mass=None):
    """Process DS agent response - NON-BLOCKING for Streamlit UI"""
    try:
        if is_initial:
            # Reset DS system
            ds_system.conversation_history = []
            ds_system.user_response = None
            # Clear belief history only for NEW queries
            if hasattr(ds_system, 'clear_belief_history'):
                ds_system.clear_belief_history()
        
        # Compute mass function for current query
        current_mass = ds_system.compute_mass_function(query_text)
        
        # Combine with previous mass if this is a followup clarification
        if previous_mass is not None and not is_initial:
            combined_mass = ds_system.combine_mass_functions(previous_mass, current_mass)
        else:
            combined_mass = current_mass
        
        # Check confidence manually WITHOUT blocking on input()
        # Get all leaf nodes
        leaf_nodes = [intent for intent in ds_system.hierarchy if ds_system.is_leaf(intent)]
        
        # Evaluate which leaves are confident
        confident_nodes, belief = ds_system.evaluate_hierarchy(leaf_nodes, combined_mass)

        # Record belief progression for explainability
        if hasattr(ds_system, 'get_belief_tracker'):
            tracker = ds_system.get_belief_tracker()
            if tracker is not None:
                turn_num = len([msg for msg in ds_system.conversation_history if msg.startswith("User:")])
                tracker.record_belief(belief, f"Turn {max(turn_num, 1)}")
        
        # Check if we have exactly one confident leaf node
        confident_leaf_nodes = [n for n in confident_nodes if ds_system.is_leaf(n[0])]
        
        if len(confident_leaf_nodes) == 1:
            # Single confident leaf - predict it
            pred_intent, confidence = confident_leaf_nodes[0]
            st.session_state.last_prediction = pred_intent
            st.session_state.last_confidence = confidence
            
            # Get natural description for the intent
            intent_desc = ""
            if hasattr(ds_system, 'intent_descriptions') and ds_system.intent_descriptions:
                intent_desc = ds_system.intent_descriptions.get(pred_intent, "")
            
            # Convert description to conversational response
            if intent_desc:
                desc_clean = intent_desc.rstrip('.').strip().lower()
                # Pattern matching for natural responses
                if desc_clean.startswith('help with'):
                    response = f"Got it! You need {desc_clean}."
                elif desc_clean.startswith('how to'):
                    action = desc_clean.replace('how to', '').strip()
                    response = f"I understand! You want to {action}."
                elif any(x in desc_clean for x in ['information about', 'details about']):
                    response = f"I see! You're looking for {desc_clean}."
                else:
                    response = f"I understand! {desc_clean.capitalize()}."
            else:
                # Fallback if no description
                response = f"I believe I can help you with {pred_intent.replace('_', ' ')}."
            
            response = _humanize_response(
                response,
                response_type="prediction",
                context={"intent": pred_intent, "description": intent_desc}
            )
            return response, False, combined_mass
        else:
            # Need clarification - use DS system's built-in clarification generator
            clarification = ds_system.generate_clarification_question(combined_mass)
            options_text = _extract_options_from_clarification(clarification)
            clarification = _humanize_response(
                clarification,
                response_type="clarification",
                context={"options": options_text}
            )
            return clarification, True, combined_mass
            
    except Exception as e:
        st.error(f"DS Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return "I encountered an error. Could you rephrase?", True, None

def generate_belief_visualization(ds_system, title="Belief Progression"):
    """Generate real-time belief visualization using BeliefVisualizer."""
    try:
        tracker = None
        if hasattr(ds_system, 'get_belief_tracker'):
            tracker = ds_system.get_belief_tracker()
        if tracker is None:
            tracker = BeliefTracker()

        history = tracker.get_history()
        if not history:
            return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            BeliefVisualizer.plot_top_intents_progression(
                belief_history=history,
                top_k=5,
                title=title,
                save_path=tmp.name,
                figsize=(10, 6)
            )
            tmp_path = tmp.name

        with open(tmp_path, "rb") as img_file:
            buf = BytesIO(img_file.read())
        os.unlink(tmp_path)
        return buf

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")
        return None

def generate_confidence_explanation(ds_system):
    """Generate confidence explanation using BeliefVisualizer."""
    try:
        tracker = None
        if hasattr(ds_system, 'get_belief_tracker'):
            tracker = ds_system.get_belief_tracker()
        if tracker is None:
            tracker = BeliefTracker()

        history = tracker.get_history()
        if not history:
            return None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            BeliefVisualizer.plot_belief_progression(
                belief_history=history,
                title="Belief Progression and Uncertainty",
                save_path=tmp.name,
                figsize=(12, 6),
                show_uncertainty=True
            )
            tmp_path = tmp.name

        with open(tmp_path, "rb") as img_file:
            buf = BytesIO(img_file.read())
        os.unlink(tmp_path)
        return buf

    except Exception as e:
        st.error(f"Confidence visualization error: {str(e)}")
        return None


def generate_confidence_explanation_interactive(ds_system):
    """Generate interactive confidence chart using Plotly."""
    try:
        tracker = None
        if hasattr(ds_system, 'get_belief_tracker'):
            tracker = ds_system.get_belief_tracker()
        if tracker is None:
            tracker = BeliefTracker()

        history = tracker.get_history()
        if not history:
            return None

        # Build grouped bar chart by turn for top intents + uncertainty
        all_intents = set()
        for belief_dict, _ in history:
            all_intents.update(belief_dict.keys())

        # Include all intents (match static plot behavior)
        top_intents = sorted(all_intents)

        records = []
        for belief_dict, label in history:
            for intent in top_intents:
                records.append({
                    "Intent": intent,
                    "Belief": belief_dict.get(intent, 0.0),
                    "Turn": label
                })

        df = pd.DataFrame(records)
        fig = px.bar(
            df,
            x="Intent",
            y="Belief",
            color="Turn",
            barmode="group",
            title="Belief Progression and Uncertainty"
        )
        fig.update_layout(
            xaxis_title="Intent",
            yaxis_title="Belief Value",
            legend_title="Turn",
            height=500,
            margin=dict(l=40, r=40, t=60, b=80)
        )
        fig.update_xaxes(tickangle=30)
        fig.update_yaxes(range=[0, 1])
        return fig
    except Exception as e:
        st.error(f"Interactive visualization error: {str(e)}")
        return None


def main():
    """Main application"""
    
    # Initialize systems ONCE with cache
    ds_system = initialize_ds_system()
    queries_df = load_study_queries()
    
    # Initialize data logger (for GitHub save)
    if 'data_logger_initialized' not in st.session_state:
        init_logger()
        st.session_state.data_logger_initialized = True
    
    # Initialize session state
    if 'current_query_index' not in st.session_state:
        st.session_state.current_query_index = 0
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'awaiting_clarification' not in st.session_state:
        st.session_state.awaiting_clarification = False
    if 'current_mass' not in st.session_state:
        st.session_state.current_mass = None
    if 'ds_system_state' not in st.session_state:
        st.session_state.ds_system_state = None
    if 'session_results' not in st.session_state:
        st.session_state.session_results = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if 'query_start_time' not in st.session_state:
        st.session_state.query_start_time = None
    if 'result_saved' not in st.session_state:
        st.session_state.result_saved = False
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = None
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'query_resolved' not in st.session_state:
        st.session_state.query_resolved = False
    if 'last_belief_plot' not in st.session_state:
        st.session_state.last_belief_plot = None
    if 'last_confidence_plot' not in st.session_state:
        st.session_state.last_confidence_plot = None
    if 'show_belief_chart' not in st.session_state:
        st.session_state.show_belief_chart = False
    if 'humanize_responses' not in st.session_state:
        st.session_state.humanize_responses = True
    if 'llm_model_name' not in st.session_state:
        st.session_state.llm_model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if 'llm_temperature' not in st.session_state:
        st.session_state.llm_temperature = float(os.getenv("TEMPERATURE", "0.6"))
    if 'llm_max_tokens' not in st.session_state:
        st.session_state.llm_max_tokens = int(os.getenv("MAX_TOKENS", "400"))
    if 'llm_warning_shown' not in st.session_state:
        st.session_state.llm_warning_shown = False
    
    # Show header
    show_header()

    # LLM warning moved to sidebar only (don't distract participants)
    if st.session_state.humanize_responses and not _llm_configured():
        if not st.session_state.llm_warning_shown:
            # Only show in sidebar, not main interface
            st.session_state.llm_warning_shown = True
    
    # Check if completed all queries
    if st.session_state.current_query_index >= len(queries_df):
        st.success(f"🎉 Completed all {len(queries_df)} queries!")
        st.balloons()
        
        # Calculate session statistics
        if st.session_state.session_results:
            completed = len(st.session_state.session_results)
            correct = sum(1 for r in st.session_state.session_results if r.get('is_correct', False))
            avg_interactions = np.mean([r.get('num_clarification_turns', 0) 
                                       for r in st.session_state.session_results])
            avg_time = np.mean([r.get('interaction_time_seconds', 0) 
                               for r in st.session_state.session_results])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Queries Completed", completed)
            with col2:
                st.metric("Accuracy", f"{100*correct/completed:.1f}%")
            with col3:
                st.metric("Avg Interactions", f"{avg_interactions:.1f}")
            with col4:
                st.metric("Avg Time (sec)", f"{avg_time:.1f}")
        
        st.markdown("---")
        
        # Final feedback form (overall experience)
        if not st.session_state.get('final_feedback_submitted', False):
            st.markdown("### 📊 Final Survey")
            st.markdown("Please share your overall experience with the system:")
            
            with st.form("final_feedback"):
                col1, col2 = st.columns(2)
                
                with col1:
                    overall_rating = st.select_slider(
                        "Overall experience rating",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        format_func=lambda x: ["😞 Poor", "😐 Fair", "🙂 Good", "😊 Very Good", "🤩 Excellent"][x-1]
                    )
                    
                    trust = st.select_slider(
                        "How much do you trust the system?",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        format_func=lambda x: "⭐" * x
                    )
                
                with col2:
                    ease_of_use = st.select_slider(
                        "Ease of use",
                        options=[1, 2, 3, 4, 5],
                        value=3,
                        format_func=lambda x: "⭐" * x
                    )
                    
                    would_recommend = st.radio(
                        "Would you recommend this system?",
                        ["Definitely", "Probably", "Maybe", "Probably Not", "Definitely Not"],
                        index=2
                    )
                
                additional_comments = st.text_area(
                    "Additional comments (optional)",
                    placeholder="What did you like? What could be improved? Any other feedback?",
                    height=100
                )
                
                submitted = st.form_submit_button("📤 Submit Final Feedback", type="primary", use_container_width=True)
                
                if submitted:
                    # Save final feedback
                    final_feedback = {
                        "overall_rating": overall_rating,
                        "trust": trust,
                        "ease_of_use": ease_of_use,
                        "would_recommend": would_recommend,
                        "additional_comments": additional_comments,
                        "session_id": st.session_state.session_id,
                        "participant_id": st.session_state.get("pid", ""),
                        "condition": st.session_state.get("cond", ""),
                        "prolific_pid": st.session_state.get("prolific_pid", ""),
                        "timestamp": datetime.datetime.now().isoformat(),
                        "num_queries_completed": len(st.session_state.session_results),
                        "accuracy": correct / completed if completed > 0 else 0,
                        "avg_clarifications": avg_interactions,
                        "avg_time_seconds": avg_time
                    }
                    
                    # Log to data logger before saving to GitHub
                    if 'data_logger' in st.session_state and st.session_state.data_logger:
                        st.session_state.data_logger.set_final_feedback(final_feedback)
                        
                        # Save to GitHub (with local fallback)
                        with st.spinner("📤 Saving session data..."):
                            save_success = save_session_to_github()
                            if save_success:
                                st.success("Session data saved successfully!")
                            else:
                                st.warning("WARNING: Data saved locally (GitHub sync may have failed)")
                    
                    # Also save local copies for redundancy
                    feedback_dir = Path("outputs/user_study/feedback")
                    feedback_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save final feedback as JSON
                    feedback_file = feedback_dir / f"session_{st.session_state.session_id}_final.json"
                    with open(feedback_file, 'w') as f:
                        json.dump(final_feedback, f, indent=2)
                    
                    # Save complete session data (results + feedback)
                    complete_data = {
                        "final_feedback": final_feedback,
                        "query_results": st.session_state.session_results
                    }
                    complete_file = feedback_dir / f"session_{st.session_state.session_id}_complete.json"
                    with open(complete_file, 'w') as f:
                        json.dump(complete_data, f, indent=2)
                    
                    st.success("Thank you! Your feedback has been saved.")
                    st.session_state.final_feedback_submitted = True
                    st.rerun()
        else:
            st.success("Final feedback already submitted. Thank you!")
        
        st.markdown("---")
        
        # Show download button for session results
        if st.session_state.session_results:
            results_df = pd.DataFrame(st.session_state.session_results)
            csv_data = results_df.to_csv(index=False)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"human_session_{st.session_state.session_id}_{timestamp}.csv"
            
            st.download_button(
                label="📥 Download Your Session Results (CSV)",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        
        # Qualtrics return button (if return URL provided)
        if st.session_state.get("return_raw") and st.session_state.get('final_feedback_submitted', False):
            st.markdown("---")
            st.markdown("### Study Complete")
            st.info("You can now return to the survey platform.")
            if st.button("🔙 Return to Survey", type="primary", use_container_width=True):
                back_to_survey(done_flag=True)
        elif not st.session_state.get("return_raw"):
            # No return URL, show restart option
            st.markdown("---")
            if st.button("🔄 Start New Session"):
                # Reset for new session
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        return
    
    # Get current query
    current_query = queries_df.iloc[st.session_state.current_query_index]
    query_text = current_query['query']
    
    # Progress indicator with stats
    progress = st.session_state.current_query_index / len(queries_df)
    st.progress(progress)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        remaining = len(queries_df) - st.session_state.current_query_index
        st.caption(f"Query {st.session_state.current_query_index + 1} of {len(queries_df)} ({remaining} remaining)")
    with col2:
        if st.session_state.session_results:
            completed = len(st.session_state.session_results)
            correct = sum(1 for r in st.session_state.session_results if r.get('is_correct', False))
            st.caption(f"Accuracy: {correct}/{completed} ({100*correct/completed:.1f}%)")
        else:
            st.caption("Tip: Press Enter = Next | Type 'why' = Explain")
    with col3:
        with st.popover("Settings"):
            st.checkbox(
                "Humanized responses (LLM)",
                value=st.session_state.humanize_responses,
                key="humanize_responses",
                help="Rewrite clarifications and explanations using OpenAI when configured."
            )
            if st.session_state.humanize_responses:
                if _llm_configured():
                    st.caption("LLM status: active (humanized responses enabled)")
                else:
                    st.caption("LLM status: unavailable (falling back to templates)")
            st.text_input(
                "LLM model",
                value=st.session_state.llm_model_name,
                key="llm_model_name",
                help="Model used for humanized responses (e.g., gpt-4o-mini)."
            )
            st.slider(
                "LLM temperature",
                min_value=0.0,
                max_value=1.2,
                value=float(st.session_state.llm_temperature),
                step=0.05,
                key="llm_temperature",
                help="Higher values make responses warmer and more varied."
            )
            st.number_input(
                "LLM max tokens",
                min_value=50,
                max_value=1000,
                value=int(st.session_state.llm_max_tokens),
                step=25,
                key="llm_max_tokens",
                help="Token budget for LLM response generation."
            )
            if st.button("Test LLM connection"):
                ok, message = _test_llm_connection()
                if ok:
                    st.success(message)
                else:
                    st.error(message)
            if st.button("Skip to end (testing)", help="Skip remaining queries for testing"):
                # Save current and jump to end
                if st.session_state.conversation_started:
                    save_query_result(current_query, ds_system)
                st.session_state.current_query_index = len(queries_df)
                st.rerun()


    
    # Auto-start query processing (no button click needed)
    if not st.session_state.conversation_started:
        st.session_state.query_start_time = datetime.datetime.now()
        # Process with DS system
        response, needs_clarification, mass = process_query(query_text, ds_system, is_initial=True)
        st.session_state.conversation_history = [
            f"User: {query_text}",
            f"Assistant: {response}"
        ]
        st.session_state.awaiting_clarification = needs_clarification
        st.session_state.query_resolved = not needs_clarification
        st.session_state.current_mass = mass
        st.session_state.conversation_started = True
        # NO rerun - continue rendering
    
    # Show query at top when conversation started
    st.markdown(f"""
    <div class="query-card">
        <strong>Customer Query #{st.session_state.current_query_index + 1}:</strong><br>
        {query_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Show conversation only if there was interaction (clarification needed)
    # For immediate predictions, skip conversation display - status message is enough
    if len(st.session_state.conversation_history) > 2 or st.session_state.awaiting_clarification:
        # Show full conversation with chat bubbles
        st.markdown("<div class=\"interaction-container\">", unsafe_allow_html=True)
        for msg in st.session_state.conversation_history:
            if msg.startswith("User:"):
                user_text = msg.replace("User: ", "").strip()
                st.markdown(
                    f'<div class="user-message">{format_bubble_text(user_text)}</div>',
                    unsafe_allow_html=True
                )
            elif msg.startswith("Assistant:"):
                assistant_text = msg.replace("Assistant: ", "").strip()
                st.markdown(
                    f'<div class="bot-message">{format_bubble_text(assistant_text)}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-message">{format_bubble_text(msg)}</div>',
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

    # Chat interface at bottom
    st.markdown("---")
    
    # Show status when resolved - feedback form will appear below
    if st.session_state.query_resolved:
        # Check if prediction is correct
        predicted_intent = st.session_state.get('last_prediction', 'unknown')
        true_intent = current_query.get('true_intent', '')
        is_correct = (predicted_intent == true_intent)
        
        st.success(f"System prediction complete: **{predicted_intent}**")
        st.info("Scroll down to validate and provide feedback")
        
        # Save result to session_results ONCE when query first resolves
        if not st.session_state.get('result_saved', False):
            save_result_to_session(current_query, ds_system, predicted_intent, is_correct)
            st.session_state.result_saved = True
        
        # Show feedback form on every render (until feedback is submitted)
        if not st.session_state.get('feedback_submitted', False):
            feedback_submitted = collect_query_feedback(
                st.session_state.current_query_index,
                current_query['query'],
                predicted_intent,
                is_correct
            )
            
            if feedback_submitted:
                # Move to next query
                st.session_state.current_query_index += 1
                # Reset conversation state
                st.session_state.conversation_started = False
                st.session_state.conversation_history = []
                st.session_state.awaiting_clarification = False
                st.session_state.query_resolved = False
                st.session_state.current_mass = None
                st.session_state.query_start_time = None
                st.session_state.result_saved = False  # Reset for next query
                st.session_state.feedback_submitted = False  # Reset for next query
                st.session_state.last_belief_plot = None
                st.session_state.last_confidence_plot = None
                st.rerun()
    elif st.session_state.awaiting_clarification:
        st.info("Please provide more information or type 'why' to understand my question.")
    
    # Chat input - for clarifications, explanations, and "why" questions
    if st.session_state.query_resolved:
        placeholder = "Type 'why' to see my reasoning, then submit feedback below..."
    else:
        placeholder = "Type your response or ask 'why?'"
    
    user_input = st.chat_input(placeholder, key=f"chat_{st.session_state.current_query_index}")
    
    if user_input:
        user_input_lower = user_input.lower().strip()
        
        # Handle "why" questions (allowed during clarification AND after resolution)
        if 'why' in user_input_lower or 'how did you' in user_input_lower or 'explain' in user_input_lower:
            st.session_state.conversation_history.append(f"User: {user_input}")
            
            # Log why question
            if 'data_logger' in st.session_state and st.session_state.data_logger:
                st.session_state.data_logger.log_why_question()
            
            if st.session_state.awaiting_clarification:
                explanation, belief_plot, confidence_plot = get_ds_explanation(ds_system, "clarification")
                explanation_type = "clarification"
            else:
                explanation, belief_plot, confidence_plot = get_ds_explanation(ds_system, "decision")
                explanation_type = "decision"
            
            # Build rich context for LLM to vary responses naturally
            conversation_context = {
                "explanation_type": explanation_type,
                "num_turns": len([m for m in st.session_state.conversation_history if m.startswith("User:")]),
                "had_clarification": len(st.session_state.conversation_history) > 2
            }
            
            explanation = _humanize_response(
                explanation,
                response_type="explanation",
                context=conversation_context
            )
            st.session_state.conversation_history.append(f"Assistant: {explanation}")
            st.session_state.last_belief_plot = belief_plot
            st.session_state.last_confidence_plot = confidence_plot
            st.rerun()
        
        # When resolved, only "why" is allowed (feedback form handles advancing)
        elif st.session_state.query_resolved:
            st.info("Query is resolved. Type 'why' to see reasoning, then use the feedback form below to continue.")
            st.rerun()
        
        # Handle clarification response (during active clarification)
        elif st.session_state.awaiting_clarification:
            st.session_state.conversation_history.append(f"User: {user_input}")
            ds_system.user_response = user_input
            # Pass previous mass to combine beliefs
            response, needs_clarification, mass = process_query(
                user_input, ds_system, is_initial=False, previous_mass=st.session_state.current_mass
            )
            st.session_state.conversation_history.append(f"Assistant: {response}")
            st.session_state.awaiting_clarification = needs_clarification
            st.session_state.query_resolved = not needs_clarification
            st.session_state.current_mass = mass
            st.rerun()

def collect_query_feedback(query_index, query_text, predicted_intent, is_correct):
    """Collect per-query feedback after resolution - includes user intent validation"""
    st.markdown("---")
    st.markdown(f"### Query Feedback")
    
    # Get true intent from query data
    queries_df = load_study_queries()
    true_intent = queries_df.iloc[query_index]['true_intent'] if query_index < len(queries_df) else 'unknown'
    
    st.markdown(f"**Your query:** \"{query_text}\"")
    st.markdown("")
    
    with st.form(f"feedback_query_{query_index}", clear_on_submit=True):
        # H7 Testing: User validates their actual intent
        st.markdown("**Which option best describes what you actually wanted?**")
        
        # Create options from system prediction and oracle label
        options = []
        option_labels = []
        
        # Add system's prediction
        if predicted_intent != 'unknown':
            system_label = f"System predicted: **{predicted_intent}**"
            options.append(predicted_intent)
            option_labels.append(system_label)
        
        # Add oracle/expected intent if different from prediction
        if true_intent != predicted_intent and true_intent != 'unknown':
            oracle_label = f"Expected intent: **{true_intent}**"
            options.append(true_intent)
            option_labels.append(oracle_label)
        
        # Add "Neither" option
        options.append("Neither/Other")
        option_labels.append("Neither of the above / Something else")
        
        # User selects their actual intent
        user_selected_intent = st.radio(
            "",  # Label already asked above
            options=options,
            format_func=lambda x: option_labels[options.index(x)],
            key=f"user_intent_{query_index}"
        )
        
        st.markdown("---")
        st.markdown("**Rate the interaction:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            clarity = st.select_slider(
                "How clear was the conversation?",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x,
                key=f"clarity_{query_index}"
            )
        
        with col2:
            confidence_rating = st.select_slider(
                "How confident are you in the result?",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: "⭐" * x,
                key=f"confidence_rating_{query_index}"
            )
        
        # Optional comment
        comment = st.text_input(
            "Any comments? (optional)",
            placeholder="e.g., 'Needed too many clarifications' or 'Worked perfectly'",
            key=f"comment_{query_index}"
        )
        
        submitted = st.form_submit_button("Submit Feedback & Continue", type="primary", use_container_width=True)
        
        if submitted:
            # Save feedback to the most recent result
            if 'session_results' in st.session_state and st.session_state.session_results:
                # Update the last result with feedback
                st.session_state.session_results[-1]['user_validated_intent'] = user_selected_intent
                st.session_state.session_results[-1]['user_agrees_with_system'] = (user_selected_intent == predicted_intent)
                st.session_state.session_results[-1]['user_agrees_with_oracle'] = (user_selected_intent == true_intent)
                st.session_state.session_results[-1]['feedback_clarity'] = clarity
                st.session_state.session_results[-1]['feedback_confidence'] = confidence_rating
                st.session_state.session_results[-1]['feedback_comment'] = comment
                st.session_state.session_results[-1]['feedback_submitted'] = True
                
                # Log to data logger
                if 'data_logger' in st.session_state and st.session_state.data_logger:
                    st.session_state.data_logger.log_query_result(st.session_state.session_results[-1])
            
            # Mark feedback as complete to prevent re-showing form
            st.session_state.feedback_submitted = True
            return True
    
    return False

def save_result_to_session(query_row, ds_system, predicted_intent, is_correct):
    """Save the completed query interaction to session results (called once per query)"""
    try:
        confidence = st.session_state.get('last_confidence', 0.0)
        
        # Count clarification turns (user messages - 1 for initial query)
        clarification_turns = len([msg for msg in st.session_state.conversation_history if msg.startswith("User:")]) - 1
        if clarification_turns < 0:
            clarification_turns = 0
        
        # Calculate interaction time
        end_time = datetime.datetime.now()
        interaction_time = (end_time - st.session_state.query_start_time).total_seconds() if st.session_state.query_start_time else 0
        
        # Create result record
        result = {
            'session_id': st.session_state.session_id,
            'query_index': st.session_state.current_query_index,
            'query_text': query_row['query'],
            'true_intent': query_row['true_intent'],
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'num_clarification_turns': clarification_turns,
            'is_correct': is_correct,
            'interaction_time_seconds': interaction_time,
            'conversation_transcript': '\n'.join(st.session_state.conversation_history),
            'timestamp': end_time.isoformat(),
            'llm_predicted_intent': query_row.get('predicted_intent', ''),
            'llm_num_interactions': query_row.get('num_interactions', 0),
            'llm_confidence': query_row.get('confidence', 0.0),
            'llm_was_correct': query_row.get('is_correct', False),
            # H7 Testing: User validation fields
            'user_validated_intent': None,
            'user_agrees_with_system': None,
            'user_agrees_with_oracle': None,
            # Feedback fields will be added after feedback collection
            'feedback_clarity': None,
            'feedback_confidence': None,
            'feedback_comment': '',
            'feedback_submitted': False
        }
        
        st.session_state.session_results.append(result)
        
    except Exception as e:
        st.error(f"Error saving query result: {str(e)}")

def save_query_result(query_row, ds_system):
    """DEPRECATED - kept for backward compatibility with skip button"""
    # This function is now split into save_result_to_session() and collect_query_feedback()
    # Only called by the "Skip to end" button in sidebar
    try:
        # Extract final prediction from conversation or session state
        final_prediction = st.session_state.get('last_prediction', 'unknown')
        confidence = st.session_state.get('last_confidence', 0.0)
        
        # Fallback: try to extract from conversation if not in session state
        if final_prediction == 'unknown':
            for msg in st.session_state.conversation_history:
                if msg.startswith("Assistant:"):
                    # Try to extract intent from **intent** format (works with or without confidence)
                    match = re.search(r'\*\*(.*?)\*\*', msg)
                    if match:
                        final_prediction = match.group(1)
                        break

        # Count clarification turns (user messages - 1 for initial query)
        clarification_turns = len([msg for msg in st.session_state.conversation_history if msg.startswith("User:")]) - 1
        if clarification_turns < 0:
            clarification_turns = 0
        
        # Calculate interaction time
        end_time = datetime.datetime.now()
        interaction_time = (end_time - st.session_state.query_start_time).total_seconds() if st.session_state.query_start_time else 0
        
        # Check correctness
        is_correct = final_prediction == query_row['true_intent']
        
        # Create result record
        result = {
            'session_id': st.session_state.session_id,
            'query_index': st.session_state.current_query_index,
            'query_text': query_row['query'],
            'true_intent': query_row['true_intent'],
            'predicted_intent': final_prediction,
            'confidence': confidence,
            'num_clarification_turns': clarification_turns,
            'is_correct': is_correct,
            'interaction_time_seconds': interaction_time,
            'conversation_transcript': '\n'.join(st.session_state.conversation_history),
            'timestamp': end_time.isoformat(),
            'llm_predicted_intent': query_row.get('predicted_intent', ''),
            'llm_num_interactions': query_row.get('num_interactions', 0),
            'llm_confidence': query_row.get('confidence', 0.0),
            'llm_was_correct': query_row.get('is_correct', False),
            # H7 Testing: User validation fields
            'user_validated_intent': None,
            'user_agrees_with_system': None,
            'user_agrees_with_oracle': None,
            # Feedback fields will be added after feedback collection
            'feedback_clarity': None,
            'feedback_confidence': None,
            'feedback_comment': '',
            'feedback_submitted': False
        }
        
        st.session_state.session_results.append(result)
        
        # Show feedback form ONLY if not already submitted
        if not st.session_state.get('feedback_submitted', False):
            feedback_submitted = collect_query_feedback(
                st.session_state.current_query_index,
                query_row['query'],
                final_prediction,
                is_correct
            )
            
            if feedback_submitted:
                # Move to next query
                st.session_state.current_query_index += 1
                # Reset conversation state
                st.session_state.conversation_started = False
                st.session_state.conversation_history = []
                st.session_state.awaiting_clarification = False
                st.session_state.query_resolved = False
                st.session_state.current_mass = None
                st.session_state.query_start_time = None
                st.session_state.feedback_form_shown = False
                st.session_state.feedback_submitted = False  # Reset for next query
                st.session_state.last_belief_plot = None
                st.session_state.last_confidence_plot = None
                st.rerun()
        
    except Exception as e:
        st.error(f"Error saving query result: {str(e)}")

def get_ds_explanation(ds_system, explanation_type):
    """Get conversational explanation as natural dialogue"""
    try:
        if not hasattr(ds_system, 'get_belief_tracker') or not ds_system.get_belief_tracker():
            return "I'm using my understanding of your words to figure out what you need.", None, None
            
        tracker = ds_system.get_belief_tracker()
        final_belief = tracker.get_final_belief()
        history = tracker.get_history()
        
        if explanation_type == "clarification":
            # Conversational explanation for asking a clarification question
            if final_belief:
                sorted_beliefs = sorted(final_belief.items(), key=lambda x: x[1], reverse=True)[:2]
                
                # Convert to natural descriptions
                options = []
                for intent, belief in sorted_beliefs:
                    if hasattr(ds_system, 'intent_descriptions') and ds_system.intent_descriptions:
                        desc = ds_system.intent_descriptions.get(intent, intent.replace('_', ' '))
                        options.append(desc.lower().rstrip('.'))
                    else:
                        options.append(intent.replace('_', ' '))
                
                if len(options) >= 2:
                    text_explanation = (
                        f"I wasn't quite sure if you meant {options[0]} or {options[1]}. "
                        "Both seemed possible based on what you said, so I wanted to check with you to make sure I got it right."
                    )
                else:
                    text_explanation = (
                        "I wanted to make sure I understood exactly what you need, "
                        "so I asked for a bit more detail."
                    )
            else:
                text_explanation = "I needed more information to understand your request better."
            
            # Don't return plots for conversational mode
            return text_explanation, None, None
        
        elif explanation_type == "decision":
            # Conversational explanation for the decision
            if final_belief:
                leaf_beliefs = {
                    intent: score
                    for intent, score in final_belief.items()
                    if ds_system.is_leaf(intent)
                }
                if leaf_beliefs:
                    top_intent = max(leaf_beliefs.keys(), key=lambda x: leaf_beliefs[x])
                else:
                    top_intent = max(final_belief.keys(), key=lambda x: final_belief[x])
                
                # Get natural description
                intent_desc = ""
                if hasattr(ds_system, 'intent_descriptions') and ds_system.intent_descriptions:
                    intent_desc = ds_system.intent_descriptions.get(top_intent, top_intent.replace('_', ' '))
                    intent_desc = intent_desc.lower().rstrip('.')
                else:
                    intent_desc = top_intent.replace('_', ' ')
                
                # Natural explanation
                text_explanation = (
                    f"Based on what you told me, it sounded like you needed help with {intent_desc}. "
                )
                
                # Add context if there was clarification
                if len(history) > 1:
                    text_explanation += (
                        "Your answer to my question helped me narrow it down and feel confident about this."
                    )
                else:
                    text_explanation += (
                        "The words you used matched this intent pretty clearly, so I felt confident about it."
                    )
            else:
                fallback_intent = st.session_state.get('last_prediction', 'your request')
                if hasattr(ds_system, 'intent_descriptions') and ds_system.intent_descriptions:
                    fallback_desc = ds_system.intent_descriptions.get(fallback_intent, fallback_intent.replace('_', ' '))
                    fallback_desc = fallback_desc.lower().rstrip('.')
                else:
                    fallback_desc = fallback_intent.replace('_', ' ') if fallback_intent else 'your request'
                
                text_explanation = f"From what you said, it seemed like you needed help with {fallback_desc}."
            
            # Don't return plots for conversational mode
            return text_explanation, None, None
    
    except Exception as e:
        return "I analyzed your message and matched it with what I know about different banking requests.", None, None
    
    return "I'm using my understanding of your words to help figure out what you need.", None, None

if __name__ == "__main__":
    main()