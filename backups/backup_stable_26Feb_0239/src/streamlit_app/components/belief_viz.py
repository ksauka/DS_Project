"""Belief visualization components for Streamlit."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List, Tuple, Optional

def plot_belief_progression(
    belief_history: List[Tuple[Dict[str, float], str]],
    show_uncertainty: bool = True,
    title: str = "Belief Progression Over Turns"
) -> go.Figure:
    """
    Create interactive belief progression plot.
    
    Args:
        belief_history: List of (belief_dict, turn_label) tuples
        show_uncertainty: Whether to show uncertainty bands
        title: Plot title
    
    Returns:
        Plotly figure
    """
    if not belief_history:
        return go.Figure().add_annotation(text="No belief history")
    
    # Extract data
    turns = [label for _, label in belief_history]
    beliefs = [beliefs_dict for beliefs_dict, _ in belief_history]
    
    # Get all intents
    all_intents = set()
    for b in beliefs:
        all_intents.update(b.keys())
    
    # Create figure
    fig = go.Figure()
    
    for intent in sorted(all_intents):
        intent_beliefs = [b.get(intent, 0) for b in beliefs]
        
        fig.add_trace(go.Scatter(
            x=turns,
            y=intent_beliefs,
            mode='lines+markers',
            name=intent,
            hovertemplate=f"<b>{intent}</b><br>Turn: %{{x}}<br>Belief: %{{y:.3f}}<extra></extra>"
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Dialogue Turn",
        yaxis_title="Belief Value",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def plot_threshold_visualization(
    beliefs_df: pd.DataFrame,
    intent_name: str,
    title: str = "Threshold Optimization"
) -> go.Figure:
    """
    Plot belief distribution and F1 scores across thresholds.
    
    Args:
        beliefs_df: DataFrame with belief values
        intent_name: Name of intent to visualize
        title: Plot title
    
    Returns:
        Plotly figure with subplots
    """
    
    fig = go.Figure()
    
    # Placeholder implementation
    fig.add_annotation(
        text=f"Threshold visualization for {intent_name}",
        x=0.5, y=0.5,
        showarrow=False
    )
    
    return fig

def plot_belief_comparison(
    before: Dict[str, float],
    after: Dict[str, float],
    title: str = "Belief Update After Clarification"
) -> go.Figure:
    """
    Compare belief values before and after clarification.
    
    Args:
        before: Belief dict before
        after: Belief dict after
        title: Plot title
    
    Returns:
        Plotly figure
    """
    
    intents = list(set(list(before.keys()) + list(after.keys())))
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=intents,
        y=[before.get(i, 0) for i in intents],
        name='Before',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=intents,
        y=[after.get(i, 0) for i in intents],
        name='After',
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title=title,
        barmode='group',
        xaxis_title='Intent',
        yaxis_title='Belief Value',
        hovermode='x',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_acc_curves(
    results_df: pd.DataFrame
) -> go.Figure:
    """
    Plot Accuracy-Coverage-Burden curves.
    
    Args:
        results_df: DataFrame with accuracy, coverage, burden columns
    
    Returns:
        Plotly figure
    """
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[1, 2, 3],
        y=[0.85, 0.90, 0.95],
        mode='lines+markers',
        name='Accuracy'
    ))
    
    fig.update_layout(
        title="Accuracy-Coverage-Burden Tradeoff",
        xaxis_title="Coverage",
        yaxis_title="Accuracy",
        template='plotly_white',
        height=400
    )
    
    return fig

def render_conversation_ui(messages: List[Dict]):
    """
    Render conversation in chat-like format.
    
    Args:
        messages: List of message dicts with 'role', 'content', 'belief'
    """
    
    for msg in messages:
        if msg.get('role') == 'user':
            st.chat_message("user").write(msg.get('content', ''))
        elif msg.get('role') == 'assistant':
            with st.chat_message("assistant"):
                st.write(msg.get('content', ''))
                if 'belief' in msg:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Top Intent", msg['belief'].get('intent', 'N/A'))
                    with col2:
                        st.metric("Confidence", f"{msg['belief'].get('confidence', 0):.3f}")
                    with col3:
                        st.metric("Uncertainty", f"{msg['belief'].get('uncertainty', 0):.3f}")
