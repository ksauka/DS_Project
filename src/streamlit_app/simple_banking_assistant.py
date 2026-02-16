"""Simple Banking Assistant - Sequential Query Processing

This implements the correct flow:
1. Show one query at a time (sequential)
2. DS agent processes query
3. If clarification needed → human responds
4. When resolved → next query
5. Repeat for all 91 queries
"""

import streamlit as st
import streamlit.components.v1 as components
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

# Add root path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ds_mass_function import DSMassFunction
from src.models.embeddings import SentenceEmbedder, IntentEmbeddings
from src.models.classifier import IntentClassifier
from src.utils.explainability import BeliefTracker, BeliefVisualizer
from config.hierarchy_loader import (
    load_hierarchy_from_json,
    load_hierarchical_intents_from_json
)
from config.threshold_loader import load_thresholds_from_json


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

    base_url = os.getenv("HICXAI_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
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
        os.getenv("HICXAI_OPENAI_MODEL", "gpt-4o-mini")
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
        os.getenv("HICXAI_OPENAI_MODEL", "gpt-4o-mini")
    )
    temperature = float(
        st.session_state.get(
            "llm_temperature",
            os.getenv("HICXAI_TEMPERATURE", "0.6")
        )
    )

    if response_type == "clarification":
        system_prompt = (
            "You rewrite clarification questions to sound natural and friendly. "
            "Preserve intent labels exactly as given and keep the options list unchanged. "
            "Do not add or remove options or change their wording."
        )
    else:
        system_prompt = (
            "You rewrite explanations to sound natural and conversational. "
            "Preserve all numbers, scores, labels, and list structure exactly. "
            "Do not change any facts."
        )

    ctx_lines = []
    if context:
        for key, value in context.items():
            if value:
                ctx_lines.append(f"- {key}: {value}")
    ctx_blob = "\n".join(ctx_lines) if ctx_lines else "(none)"

    user_prompt = (
        "Rewrite the following response for a human user. "
        "Keep all factual content and formatting.\n\n"
        f"Context:\n{ctx_blob}\n\n"
        f"Original Response:\n{text}\n\n"
        "Return only the rewritten response."
    )

    try:
        max_tokens = int(
            st.session_state.get(
                "llm_max_tokens",
                int(os.getenv("HICXAI_MAX_TOKENS", "400"))
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
    page_title="Clarification Asking Banking Assistant", 
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
        
        # Load trained classifier - same path as LLM simulation
        classifier_path = 'experiments/banking77/banking77_logistic_model.pkl'
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
            enable_belief_tracking=True  # Enable for real explainability
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
    # Fallback visible header (in case HTML/CSS is suppressed)
    st.title("Clarification Asking Banking Assistant")
    st.markdown("I will process each query and ask clarifying questions if needed.")

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
        padding: 20px; border-radius: 10px; margin-bottom: 30px;
        color: white; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-container">
        <h2 style="margin: 0;">Clarification Asking Banking Assistant</h2>
        <p style="margin: 10px 0 5px 0; opacity: 0.9; line-height: 1.4;">
            This banking assistant responds to queries about banking transactions.
            I will process each query and ask clarifying questions if needed.
        </p>
        <p style="margin: 5px 0 0 0; opacity: 0.8; font-size: 0.9em; line-height: 1.3;">
            <strong>How it works:</strong> If I understand your query, I'll tell you what I understood.
            If I'm unsure, I'll ask clarification questions to improve my confidence.
            You can also ask me to explain my reasoning at any time.
        </p>
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
            response = f"I understand! You want help with: **{pred_intent}** (Confidence: {confidence:.3f})"
            return response, False, combined_mass
        else:
            # Need clarification - use DS system's built-in clarification generator
            # This uses the EXACT same logic as the LLM simulation in STEP 3
            clarification = ds_system.generate_clarification_question(combined_mass)
            clarification += " (You are free to type what you mean exactly if these examples do not resonate with your intent.)"
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


def _render_uncertainty_tree(ds_system):
    """Render a simple indented tree for the final belief state."""
    tracker = None
    if hasattr(ds_system, 'get_belief_tracker'):
        tracker = ds_system.get_belief_tracker()
    if tracker is None:
        st.info("Belief history not available for tree view.")
        return

    final_belief = tracker.get_final_belief()
    if not final_belief:
        st.info("No belief values available to render the tree.")
        return

    hierarchy = ds_system.hierarchy
    all_children = {child for children in hierarchy.values() for child in children}
    roots = [node for node in hierarchy.keys() if node not in all_children]

    if not roots:
        roots = list(hierarchy.keys())

    def render_node(node: str, depth: int = 0):
        belief = final_belief.get(node, 0.0)
        indent_px = depth * 18
        st.markdown(
            f"<div style='margin-left:{indent_px}px'>- {node}: {belief:.3f}</div>",
            unsafe_allow_html=True
        )
        for child in hierarchy.get(node, []):
            render_node(child, depth + 1)

    for root in sorted(roots):
        render_node(root, 0)




def generate_uncertainty_vis_html(ds_system, turn_index: int) -> Optional[str]:
        """Generate an interactive vis-network HTML graph for a selected belief turn."""
        tracker = None
        if hasattr(ds_system, 'get_belief_tracker'):
                tracker = ds_system.get_belief_tracker()
        if tracker is None:
                return None

        history = tracker.get_history()
        if not history:
                return None

        turn_index = max(0, min(turn_index, len(history) - 1))
        belief_dict, _ = history[turn_index]
        hierarchy = ds_system.hierarchy

        # Build parent map for path highlighting
        parent_map = {}
        for parent, children in hierarchy.items():
            for child in children:
                parent_map[child] = parent

        # Identify top leaf intent and its path to root
        leaf_beliefs = {
            intent: score
            for intent, score in belief_dict.items()
            if ds_system.is_leaf(intent)
        }
        top_leaf = None
        if leaf_beliefs:
            top_leaf = max(leaf_beliefs.keys(), key=lambda x: leaf_beliefs[x])

        highlight_nodes = set()
        highlight_edges = set()
        if top_leaf:
            current = top_leaf
            highlight_nodes.add(current)
            while current in parent_map:
                parent = parent_map[current]
                highlight_nodes.add(parent)
                highlight_edges.add((parent, current))
                current = parent

        # Build nodes with belief-based size and color
        nodes = []
        for intent, belief in belief_dict.items():
            size = 12 + (belief * 30)
            if intent in highlight_nodes:
                color = "rgba(230, 126, 34, 0.95)"
                font = {"size": 12, "color": "#2c3e50", "bold": True}
                border_width = 2
            else:
                color = f"rgba(52, 152, 219, {0.3 + 0.7 * belief:.2f})"
                font = {"size": 12, "color": "#2c3e50"}
                border_width = 1

            label = f"{intent}\n{belief:.3f}"
            nodes.append({
                "id": intent,
                "label": label,
                "value": belief,
                "size": size,
                "color": color,
                "font": font,
                "borderWidth": border_width
            })

        # Ensure parents exist as nodes even if not in beliefs
        for parent in hierarchy.keys():
            if parent not in belief_dict:
                if parent in highlight_nodes:
                    color = "rgba(230, 126, 34, 0.95)"
                    font = {"size": 12, "color": "#2c3e50", "bold": True}
                    border_width = 2
                else:
                    color = "rgba(149, 165, 166, 0.4)"
                    font = {"size": 12, "color": "#2c3e50"}
                    border_width = 1
                nodes.append({
                    "id": parent,
                    "label": parent,
                    "value": 0.0,
                    "size": 10,
                    "color": color,
                    "font": font,
                    "borderWidth": border_width
                })

        # Build edges
        edges = []
        for parent, children in hierarchy.items():
            for child in children:
                if (parent, child) in highlight_edges:
                    edges.append({
                        "from": parent,
                        "to": child,
                        "color": {"color": "#e67e22"},
                        "width": 2
                    })
                else:
                    edges.append({"from": parent, "to": child})

        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        #network {{ width: 100%; height: 600px; border: 1px solid #e1e8ed; border-radius: 8px; pointer-events: auto; }}
    </style>
</head>
<body>
    <div id="network"></div>
    <script>
        const nodes = new vis.DataSet({nodes_json});
        const edges = new vis.DataSet({edges_json});
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            layout: {{ hierarchical: {{ enabled: true, direction: 'UD', nodeSpacing: 200, levelSeparation: 120 }} }},
            physics: {{ enabled: false }},
            interaction: {{
                dragNodes: true,
                zoomView: true,
                dragView: true,
                selectable: true,
                hover: true,
                navigationButtons: true,
                keyboard: true
            }},
            autoResize: true,
            nodes: {{ shape: 'dot', font: {{ size: 12 }}, borderWidth: 1 }},
            edges: {{ arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }}, smooth: {{ type: 'cubicBezier' }} }}
        }};
        const network = new vis.Network(container, data, options);
        window.addEventListener('resize', () => network.redraw());
    </script>
</body>
</html>
"""
        return html

def main():
    """Main application"""
    
    # Initialize systems ONCE with cache
    ds_system = initialize_ds_system()
    queries_df = load_study_queries()
    
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
        st.session_state.llm_model_name = os.getenv("HICXAI_OPENAI_MODEL", "gpt-4o-mini")
    if 'llm_temperature' not in st.session_state:
        st.session_state.llm_temperature = float(os.getenv("HICXAI_TEMPERATURE", "0.6"))
    if 'llm_max_tokens' not in st.session_state:
        st.session_state.llm_max_tokens = int(os.getenv("HICXAI_MAX_TOKENS", "400"))
    if 'llm_warning_shown' not in st.session_state:
        st.session_state.llm_warning_shown = False
    
    # Show header
    show_header()

    if st.session_state.humanize_responses and not _llm_configured():
        if not st.session_state.llm_warning_shown:
            st.warning("LLM humanization is enabled, but OPENAI_API_KEY is not set. Falling back to template responses.")
            st.session_state.llm_warning_shown = True
    
    # Check if completed all queries
    if st.session_state.current_query_index >= len(queries_df):
        st.success(f" Completed all {len(queries_df)} queries!")
        st.balloons()
        
        # Show download button for session results
        if st.session_state.session_results:
            results_df = pd.DataFrame(st.session_state.session_results)
            csv_data = results_df.to_csv(index=False)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"human_session_{st.session_state.session_id}_{timestamp}.csv"
            
            st.download_button(
                label="Download Your Session Results",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            ) 
            st.info(f"Your session results contain {len(st.session_state.session_results)} completed queries.")
        
        if st.button("Start New Session"):
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
    
    # Show compact result if resolved immediately (no clarification needed)
    if st.session_state.query_resolved and len(st.session_state.conversation_history) <= 2:
        # Simple immediate resolution - show compact view
        for msg in st.session_state.conversation_history:
            if msg.startswith("Assistant:"):
                assistant_text = msg.replace("Assistant: ", "").strip()
                st.markdown(
                    f'<div class="bot-message">{format_bubble_text(assistant_text)}</div>',
                    unsafe_allow_html=True
                )
    else:
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

    # Display belief visualizations if available
    if st.session_state.get('last_belief_plot'):
        st.markdown("### Belief Progression")
        if st.session_state.get('last_belief_plot'):
            st.image(st.session_state['last_belief_plot'], caption="Top 5 Intents Over Time", width=520)
        st.divider()

        with st.expander("View large belief chart", expanded=False):
            if st.session_state.get('last_belief_plot'):
                st.image(st.session_state['last_belief_plot'], caption="Top 5 Intents Over Time (Large)", width=1100)

        with st.expander("View uncertainty as tree", expanded=False):
            _render_uncertainty_tree(ds_system)

        with st.expander("View uncertainty graph (vis-network)", expanded=False):
            tracker = ds_system.get_belief_tracker() if hasattr(ds_system, 'get_belief_tracker') else None
            history = tracker.get_history() if tracker is not None else []
            if history:
                turn_labels = [label for _, label in history]
                selected_label = st.selectbox(
                    "Select turn (vis)",
                    options=turn_labels,
                    index=len(turn_labels) - 1,
                    key="vis_turn_select"
                )
                turn_index = turn_labels.index(selected_label)
                html = generate_uncertainty_vis_html(ds_system, turn_index)
                if html:
                    components.html(html, height=650, scrolling=True)
            else:
                st.info("Belief history not available for vis-network view.")
    
    # Display belief visualization after clarification
    if len(st.session_state.conversation_history) > 2 and not st.session_state.awaiting_clarification:
        if st.session_state.get('show_belief_chart'):
            with st.expander("View Belief Progression Chart", expanded=False):
                belief_viz = generate_belief_visualization(ds_system, "Belief Evolution")
                if belief_viz:
                    st.image(belief_viz, width=900)

    

    # Chat interface at bottom
    st.markdown("---")
    
    # Show status and Next button when resolved
    if st.session_state.query_resolved:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("Query resolved! Press Enter/type 'next' or ask 'why' to see reasoning.")
        with col2:
            if st.button("Next", use_container_width=True, type="primary", key=f"next_btn_{st.session_state.current_query_index}"):
                save_query_result(current_query, ds_system)
                st.session_state.current_query_index += 1
                # Reset state for next query
                st.session_state.conversation_history = []
                st.session_state.awaiting_clarification = False
                st.session_state.conversation_started = False
                st.session_state.query_resolved = False
                st.session_state.last_belief_plot = None
                st.session_state.last_confidence_plot = None
                st.session_state.show_belief_chart = False
                st.session_state.current_mass = None
                st.session_state.query_start_time = None
                st.rerun()
    elif st.session_state.awaiting_clarification:
        st.info("Please provide more information or type 'why' to understand my question.")
    
    # Chat input - optimized placeholder based on state
    if st.session_state.query_resolved:
        placeholder = "Press Enter for next query, or type 'why' to explain..."
    else:
        placeholder = "Type your response or ask 'why?'"
    
    user_input = st.chat_input(placeholder, key=f"chat_{st.session_state.current_query_index}")
    
    if user_input:
        user_input_lower = user_input.lower().strip()
        
        # Handle "next" or empty enter when resolved (fast advance)
        if st.session_state.query_resolved and (not user_input.strip() or user_input_lower in ['next', 'next query', 'continue', 'n']):
            save_query_result(current_query, ds_system)
            st.session_state.current_query_index += 1
            # Reset state for next query
            st.session_state.conversation_history = []
            st.session_state.awaiting_clarification = False
            st.session_state.conversation_started = False
            st.session_state.query_resolved = False
            st.session_state.current_mass = None
            st.session_state.query_start_time = None
            st.rerun()
        
        # Handle "why" questions
        elif 'why' in user_input_lower or 'how did you' in user_input_lower or 'explain' in user_input_lower:
            st.session_state.conversation_history.append(f"User: {user_input}")
            if st.session_state.awaiting_clarification:
                explanation, belief_plot, confidence_plot = get_ds_explanation(ds_system, "clarification")
            else:
                explanation, belief_plot, confidence_plot = get_ds_explanation(ds_system, "decision")
            explanation = _humanize_response(
                explanation,
                response_type="explanation",
                context={"explanation_type": "clarification" if st.session_state.awaiting_clarification else "decision"}
            )
            st.session_state.conversation_history.append(f"Assistant: {explanation}")
            st.session_state.last_belief_plot = belief_plot
            st.session_state.last_confidence_plot = confidence_plot
            st.rerun()
        
        # Handle clarification response
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
        
        # Query already resolved
        elif st.session_state.query_resolved:
            st.session_state.conversation_history.append(f"User: {user_input}")
            st.session_state.conversation_history.append(
                "Assistant: I've already resolved this query. Type 'next' to continue or 'why' to see my reasoning."
            )
            st.rerun()
        
        # Shouldn't reach here, but handle gracefully
        else:
            st.session_state.conversation_history.append(f"User: {user_input}")
            st.session_state.conversation_history.append(
                "Assistant: I'm processing your request..."
            )
            st.rerun()

def save_query_result(query_row, ds_system):
    """Save the completed query interaction to session results"""
    try:
        # Extract final prediction from conversation
        final_prediction = "unknown"
        confidence = 0.0
        # Look for prediction in conversation history
        for msg in st.session_state.conversation_history:
            if msg.startswith("Assistant:") and "I understand!" in msg:
                # Extract prediction from "I understand! You want help with: **intent** (Confidence: 0.xxx)"
                match = re.search(r'\*\*(.*?)\*\*\s*\(Confidence:\s*([0-9.]+)\)', msg)
                if match:
                    final_prediction = match.group(1)
                    confidence = float(match.group(2))
                break

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
            'predicted_intent': final_prediction,
            'confidence': confidence,
            'num_clarification_turns': clarification_turns,
            'is_correct': final_prediction == query_row['true_intent'],
            'interaction_time_seconds': interaction_time,
            'conversation_transcript': '\n'.join(st.session_state.conversation_history),
            'timestamp': end_time.isoformat(),
            'llm_predicted_intent': query_row.get('predicted_intent', ''),
            'llm_num_interactions': query_row.get('num_interactions', 0),
            'llm_confidence': query_row.get('confidence', 0.0),
            'llm_was_correct': query_row.get('is_correct', False)
        }
        
        st.session_state.session_results.append(result)
        
    except Exception as e:
        st.error(f"Error saving query result: {str(e)}")

def get_ds_explanation(ds_system, explanation_type):
    """Get real explainability with visual components"""
    try:
        if not hasattr(ds_system, 'get_belief_tracker') or not ds_system.get_belief_tracker():
            return "Belief tracking not available. Using Dempster-Shafer theory for reasoning.", None, None
            
        tracker = ds_system.get_belief_tracker()
        final_belief = tracker.get_final_belief()
        history = tracker.get_history()
        
        if explanation_type == "clarification":
            # Get actual reasoning for clarification
            text_explanation = "I asked for clarification because:"
            
            if final_belief:
                # Show uncertainty analysis
                sorted_beliefs = sorted(final_belief.items(), key=lambda x: x[1], reverse=True)[:3]
                intent_summaries = []
                for intent, belief in sorted_beliefs:
                    threshold = getattr(ds_system, 'get_threshold', lambda x: 0.7)(intent)
                    status = "above" if belief >= threshold else "below"
                    intent_summaries.append(f"{intent} ({belief:.3f}, {status} threshold {threshold:.3f})")
                joined = "; ".join(intent_summaries)
                text_explanation += (
                    " I was seeing multiple close options, so I needed more detail. "
                    f"Here were the top candidates: {joined}."
                )
                
                # Generate belief visualization
                belief_plot = generate_belief_visualization(ds_system, "Why I Asked for Clarification")
                confidence_plot = generate_confidence_explanation(ds_system)
                
                return text_explanation, belief_plot, confidence_plot
            
            return text_explanation, None, None
        
        elif explanation_type == "decision":
            # Get actual reasoning for decision
            text_explanation = "I made this decision because "
            
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
                confidence = final_belief[top_intent]
                threshold = getattr(ds_system, 'get_threshold', lambda x: 0.7)(top_intent)
                status = "confident" if confidence >= threshold else "uncertain"
                text_explanation += (
                    f"the evidence most strongly supported {top_intent}. "
                    f"The belief score was {confidence:.3f}, which is {status} relative to the "
                    f"required threshold of {threshold:.3f}."
                )
                if len(history) > 1:
                    initial_belief = history[0][0].get(top_intent, 0.0)
                    improvement = confidence - initial_belief
                    text_explanation += (
                        f" After your clarification, the belief improved by {improvement:.3f}, "
                        "which helped confirm the decision."
                    )
                
                # Generate visualizations
                belief_plot = generate_belief_visualization(ds_system, "Decision Reasoning")
                confidence_plot = generate_confidence_explanation(ds_system)
                
                return text_explanation, belief_plot, confidence_plot
            
            fallback_intent = st.session_state.get('last_prediction')
            fallback_conf = st.session_state.get('last_confidence')
            if fallback_intent is not None:
                text_explanation += f"\n\n**Final Decision: {fallback_intent}**"
            if fallback_conf is not None:
                text_explanation += f"\n- **Confidence**: {fallback_conf:.3f}"
            text_explanation += "\n- **Note**: Belief history was not available for detailed reasoning."
            return text_explanation, None, None
    
    except Exception as e:
        return f"Error generating explanation: {str(e)}", None, None
    
    return "Using Dempster-Shafer theory for banking intent reasoning.", None, None

if __name__ == "__main__":
    main()