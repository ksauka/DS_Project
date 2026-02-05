"""Simple Banking Assistant - Sequential Query Processing

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

# Page config
st.set_page_config(
    page_title="Clarification Asking Banking Assistant", 
    page_icon="🏦", 
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
    st.title("🏦 Clarification Asking Banking Assistant")
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
        <h2 style="margin: 0;">🏦 Clarification Asking Banking Assistant</h2>
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
    
    # Show header
    show_header()
    
    # Check if completed all queries
    if st.session_state.current_query_index >= len(queries_df):
        st.success(f"🎉 Completed all {len(queries_df)} queries!")
        st.balloons()
        
        # Show download button for session results
        if st.session_state.session_results:
            results_df = pd.DataFrame(st.session_state.session_results)
            csv_data = results_df.to_csv(index=False)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"human_session_{st.session_state.session_id}_{timestamp}.csv"
            
            st.download_button(
                label="💾 Download Your Session Results",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
            
            st.info(f"Your session results contain {len(st.session_state.session_results)} completed queries.")
        
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
        st.caption(f"📊 Query {st.session_state.current_query_index + 1} of {len(queries_df)} ({remaining} remaining)")
    with col2:
        if st.session_state.session_results:
            completed = len(st.session_state.session_results)
            correct = sum(1 for r in st.session_state.session_results if r.get('is_correct', False))
            st.caption(f"✅ Accuracy: {correct}/{completed} ({100*correct/completed:.1f}%)")
        else:
            st.caption("⌨️ Tip: Press Enter = Next | Type 'why' = Explain")
    with col3:
        with st.popover("⚙️"):
            if st.button("⏩ Skip to end (testing)", help="Skip remaining queries for testing"):
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
            f"💬 {query_text}",
            f"🏦 Banking Assistant: {response}"
        ]
        st.session_state.awaiting_clarification = needs_clarification
        st.session_state.query_resolved = not needs_clarification
        st.session_state.current_mass = mass
        st.session_state.conversation_started = True
        # NO rerun - continue rendering
    
    # Show query at top when conversation started
    st.markdown(f"""
    <div class="query-card">
        <strong>🎯 Customer Query #{st.session_state.current_query_index + 1}:</strong><br>
        {query_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Show compact result if resolved immediately (no clarification needed)
    if st.session_state.query_resolved and len(st.session_state.conversation_history) <= 2:
        # Simple immediate resolution - show compact view
        for msg in st.session_state.conversation_history:
            if msg.startswith("🏦 Banking Assistant:"):
                assistant_text = msg.replace("🏦 Banking Assistant: ", "").strip()
                st.markdown(
                    f'<div class="bot-message">🏦 {format_bubble_text(assistant_text)}</div>',
                    unsafe_allow_html=True
                )
    else:
        # Show full conversation with chat bubbles
        st.markdown("<div class=\"interaction-container\">", unsafe_allow_html=True)
        for msg in st.session_state.conversation_history:
            if msg.startswith("💬"):
                user_text = msg.replace("💬 ", "").strip()
                st.markdown(
                    f'<div class="user-message">{format_bubble_text(user_text)}</div>',
                    unsafe_allow_html=True
                )
            elif msg.startswith("🏦 Banking Assistant:"):
                assistant_text = msg.replace("🏦 Banking Assistant: ", "").strip()
                st.markdown(
                    f'<div class="bot-message">🏦 Banking Assistant: {format_bubble_text(assistant_text)}</div>',
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
        st.markdown("### 📊 Belief Progression")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.get('last_belief_plot'):
                st.image(st.session_state['last_belief_plot'], caption="Top 5 Intents Over Time", use_column_width=True)
        with col2:
            if st.session_state.get('last_confidence_plot'):
                st.image(st.session_state['last_confidence_plot'], caption="Uncertainty Reduction", use_column_width=True)
        st.divider()
    
    # Display belief visualization after clarification
    if len(st.session_state.conversation_history) > 2 and not st.session_state.awaiting_clarification:
        if st.session_state.get('show_belief_chart'):
            with st.expander("📈 View Belief Progression Chart", expanded=False):
                belief_viz = generate_belief_visualization(ds_system, "Belief Evolution")
                if belief_viz:
                    st.image(belief_viz, use_column_width=True)

    

    # Chat interface at bottom
    st.markdown("---")
    
    # Show status and Next button when resolved
    if st.session_state.query_resolved:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success("✅ Query resolved! Press Enter/type 'next' or ask 'why' to see reasoning.")
        with col2:
            if st.button("➡️ Next", use_container_width=True, type="primary", key=f"next_btn_{st.session_state.current_query_index}"):
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
        st.info("💭 Please provide more information or type 'why' to understand my question.")
    
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
            st.session_state.conversation_history.append(f"💬 {user_input}")
            if st.session_state.awaiting_clarification:
                explanation, belief_plot, confidence_plot = get_ds_explanation(ds_system, "clarification")
            else:
                explanation, belief_plot, confidence_plot = get_ds_explanation(ds_system, "decision")
            st.session_state.conversation_history.append(f"🏦 Banking Assistant: {explanation}")
            st.session_state.last_belief_plot = belief_plot
            st.session_state.last_confidence_plot = confidence_plot
            st.rerun()
        
        # Handle clarification response
        elif st.session_state.awaiting_clarification:
            st.session_state.conversation_history.append(f"💬 {user_input}")
            ds_system.user_response = user_input
            # Pass previous mass to combine beliefs
            response, needs_clarification, mass = process_query(
                user_input, ds_system, is_initial=False, previous_mass=st.session_state.current_mass
            )
            st.session_state.conversation_history.append(f"🏦 Banking Assistant: {response}")
            st.session_state.awaiting_clarification = needs_clarification
            st.session_state.query_resolved = not needs_clarification
            st.session_state.current_mass = mass
            st.rerun()
        
        # Query already resolved
        elif st.session_state.query_resolved:
            st.session_state.conversation_history.append(f"💬 {user_input}")
            st.session_state.conversation_history.append(
                f"🏦 Banking Assistant: I've already resolved this query. Type 'next' to continue or 'why' to see my reasoning."
            )
            st.rerun()
        
        # Shouldn't reach here, but handle gracefully
        else:
            st.session_state.conversation_history.append(f"💬 {user_input}")
            st.session_state.conversation_history.append(
                f"🏦 Banking Assistant: I'm processing your request..."
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
            if msg.startswith("🏦 Banking Assistant:") and "I understand!" in msg:
                # Extract prediction from "I understand! You want help with: **intent** (Confidence: 0.xxx)"
                match = re.search(r'\*\*(.*?)\*\*\s*\(Confidence:\s*([0-9.]+)\)', msg)
                if match:
                    final_prediction = match.group(1)
                    confidence = float(match.group(2))
                break
        
        # Count clarification turns (user messages - 1 for initial query)
        clarification_turns = len([msg for msg in st.session_state.conversation_history if msg.startswith("💬")]) - 1
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
                text_explanation += "\n\n**Top competing intents:**\n"
                for intent, belief in sorted_beliefs:
                    threshold = getattr(ds_system, 'get_threshold', lambda x: 0.7)(intent)
                    status = "✅ Above threshold" if belief >= threshold else "❌ Below threshold"
                    text_explanation += f"- **{intent}**: {belief:.3f} ({status})\n"
                
                # Generate belief visualization
                belief_plot = generate_belief_visualization(ds_system, "Why I Asked for Clarification")
                confidence_plot = generate_confidence_explanation(ds_system)
                
                return text_explanation, belief_plot, confidence_plot
            
            return text_explanation, None, None
        
        elif explanation_type == "decision":
            # Get actual reasoning for decision
            text_explanation = "I made this decision because:\n"
            
            if final_belief:
                top_intent = max(final_belief.keys(), key=lambda x: final_belief[x])
                confidence = final_belief[top_intent]
                threshold = getattr(ds_system, 'get_threshold', lambda x: 0.7)(top_intent)
                
                text_explanation += f"\n**Final Decision: {top_intent}**\n"
                text_explanation += f"- **Belief Score**: {confidence:.3f}\n"
                text_explanation += f"- **Required Threshold**: {threshold:.3f}\n"
                text_explanation += f"- **Status**: {'✅ Confident' if confidence >= threshold else '⚠️ Uncertain'}"
                
                if len(history) > 1:
                    initial_belief = history[0][0].get(top_intent, 0.0)
                    improvement = confidence - initial_belief
                    text_explanation += f"\n- **Belief Improvement**: +{improvement:.3f} through clarification"
                
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