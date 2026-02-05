"""Output formatting utilities."""

import streamlit as st
from typing import Dict, List
import pandas as pd

def format_belief_for_display(belief: Dict[str, float], top_k: int = 5) -> str:
    """
    Format belief values for display.
    
    Args:
        belief: Belief dictionary
        top_k: Show top K intents
    
    Returns:
        Formatted string
    """
    
    if not belief:
        return "No belief values"
    
    # Sort by belief value
    sorted_beliefs = sorted(
        belief.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]
    
    result = []
    for intent, value in sorted_beliefs:
        bar_length = int(value * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        result.append(f"{intent:20s} {bar} {value:.3f}")
    
    return "\n".join(result)

def format_confidence(confidence: float) -> str:
    """
    Format confidence value as readable string.
    
    Args:
        confidence: Confidence value (0-1)
    
    Returns:
        Formatted string with emoji
    """
    
    if confidence >= 0.9:
        return f"🟢 Very High ({confidence:.1%})"
    elif confidence >= 0.7:
        return f"🟡 High ({confidence:.1%})"
    elif confidence >= 0.5:
        return f"🟠 Medium ({confidence:.1%})"
    elif confidence >= 0.3:
        return f"🟡 Low ({confidence:.1%})"
    else:
        return f"🔴 Very Low ({confidence:.1%})"

def format_uncertainty(uncertainty: float) -> str:
    """
    Format uncertainty as readable string.
    
    Args:
        uncertainty: Uncertainty value (0-1)
    
    Returns:
        Formatted string with emoji
    """
    
    if uncertainty <= 0.1:
        return f"🟢 Very Low ({uncertainty:.3f})"
    elif uncertainty <= 0.3:
        return f"🟡 Low ({uncertainty:.3f})"
    elif uncertainty <= 0.5:
        return f"🟠 Medium ({uncertainty:.3f})"
    else:
        return f"🔴 High ({uncertainty:.3f})"

def format_metrics_table(metrics: Dict) -> pd.DataFrame:
    """
    Format metrics dictionary as DataFrame.
    
    Args:
        metrics: Metrics dictionary
    
    Returns:
        Pandas DataFrame
    """
    
    data = []
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            value = f"{metric_value:.3f}" if isinstance(metric_value, float) else str(metric_value)
        else:
            value = str(metric_value)
        
        data.append({"Metric": metric_name, "Value": value})
    
    return pd.DataFrame(data)

def format_conversation_history(history: List[Dict]) -> str:
    """
    Format conversation history as readable text.
    
    Args:
        history: List of message dicts
    
    Returns:
        Formatted string
    """
    
    lines = []
    turn_number = 1
    
    for msg in history:
        if msg.get('role') == 'user':
            lines.append(f"\n**Turn {turn_number}:**")
            lines.append(f"👤 User: {msg.get('content', '')}")
        elif msg.get('role') == 'assistant':
            lines.append(f"🤖 Agent: {msg.get('content', '')}")
            turn_number += 1
    
    return "\n".join(lines)

def format_results_summary(results: Dict) -> str:
    """
    Format results as summary string.
    
    Args:
        results: Results dictionary
    
    Returns:
        Formatted summary
    """
    
    summary = []
    summary.append("### Results Summary\n")
    
    if 'accuracy' in results:
        summary.append(f"**Accuracy:** {results['accuracy']:.1%}")
    
    if 'f1_score' in results:
        summary.append(f"**F1 Score:** {results['f1_score']:.3f}")
    
    if 'avg_confidence' in results:
        summary.append(f"**Avg Confidence:** {results['avg_confidence']:.3f}")
    
    if 'coverage' in results:
        summary.append(f"**Coverage:** {results['coverage']:.1%}")
    
    return "\n".join(summary)

def format_error_message(error: str, context: str = "") -> str:
    """
    Format error message for display.
    
    Args:
        error: Error message
        context: Additional context
    
    Returns:
        Formatted error string
    """
    
    message = f"❌ {error}"
    
    if context:
        message += f"\n\n**Context:** {context}"
    
    return message

def format_success_message(message: str, icon: str = "✅") -> str:
    """
    Format success message for display.
    
    Args:
        message: Success message
        icon: Icon to use
    
    Returns:
        Formatted success string
    """
    
    return f"{icon} {message}"
