"""Customer Service Assistant - Sequential Query Processing

"""

# Standard library imports
import datetime
import gc
import html
import json
import os
import random
import re
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import (
    parse_qsl,
    unquote,
    urlencode,
    urlparse,
    urlunparse,
)

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env (for local development)
load_dotenv()

# Local application imports (sys.path configured by app_main_hicxai.py)
try:
    from config.hierarchy_loader import (
        load_hierarchical_intents_from_json,
        load_hierarchy_from_json,
    )
    from config.threshold_loader import load_thresholds_from_json
    from src.models.classifier import IntentClassifier
    from src.models.ds_mass_function import DSMassFunction
    from src.models.embeddings import IntentEmbeddings, SentenceEmbedder
    from src.utils.data_logger import init_logger, save_session_to_github
    from src.utils.explainability import BeliefTracker, BeliefVisualizer
except ImportError:
    # Fallback: try relative imports if absolute imports fail
    from ...config.hierarchy_loader import (
        load_hierarchical_intents_from_json,
        load_hierarchy_from_json,
    )
    from ...config.threshold_loader import load_thresholds_from_json
    from ...src.models.classifier import IntentClassifier
    from ...src.models.ds_mass_function import DSMassFunction
    from ...src.models.embeddings import IntentEmbeddings, SentenceEmbedder
    from ...src.utils.data_logger import init_logger, save_session_to_github
    from ...src.utils.explainability import BeliefTracker, BeliefVisualizer


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
    # Use parse_qsl like anthrokit
    q = dict(parse_qsl(p.query, keep_blank_values=True))
    
    # Append parameters if missing (for data linkage in Qualtrics)
    prolific_pid_ss = st.session_state.get("prolific_pid", "")
    # KEY for data linkage
    session_id_ss = st.session_state.get("session_id", "")
    
    if "PROLIFIC_PID" not in q and prolific_pid_ss:
        q["PROLIFIC_PID"] = prolific_pid_ss
    if "pid" not in q and st.session_state.get("pid"):
        q["pid"] = st.session_state.pid
    if "cond" not in q and st.session_state.get("cond"):
        q["cond"] = st.session_state.cond
    if "session_id" not in q and session_id_ss:
        # Pass session_id back for data linkage
        q["session_id"] = session_id_ss
    if "done" not in q:
        q["done"] = "1" if done else "0"
    
    return urlunparse(p._replace(query=urlencode(q, doseq=True)))

def back_to_survey(done_flag=True):
    """Redirect back to Qualtrics/Prolific survey after study completion"""
    if st.session_state.get("_returned", False):
        return
    final = _build_final_return(done=done_flag)
    if not final:
        msg = (
            "WARNING: Return link missing or invalid. Please close this "
            "window and return to the survey manually."
        )
        st.warning(msg)
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


def _get_api_key() -> Optional[str]:
    """Get OpenAI API key from env or Streamlit secrets."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = None
    return api_key


def _get_openai_client():
    """Get an OpenAI client if API key is configured."""
    api_key = _get_api_key()
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
    return bool(_get_api_key())


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


def _extract_raw_options(text: str) -> list:
    """Extract the raw intent option names from a DS clarification string.

    E.g. "There are a few things: (['card_linking', 'getting_spare_card']). Could you clarify?"
    → ['card_linking', 'getting_spare_card']
    """
    import re
    # Match content inside ([ ... ]) or [ ... ]
    m = re.search(r"\[([^\]]+)\]", text)
    if not m:
        return []
    raw = m.group(1)
    # Extract individual quoted strings or bare words
    items = re.findall(r"['\"]?([a-zA-Z0-9_]+)['\"]?", raw)
    return [i for i in items if i]


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
    temperature = float(
        st.session_state.get(
            "llm_temperature",
            os.getenv("TEMPERATURE", "0.6")
        )
    )

    if response_type == "clarification":
        # Extract the intent options before sending anything to the LLM.
        # The LLM only writes the intro sentence; we always append the options
        # ourselves so they can never be dropped or rewritten.
        options = _extract_raw_options(text)

        if not options:
            # CASE 3: no confident nodes — DS asked user to rephrase, no option list.
            # Just rewrite the rephrase request naturally without adding an options list.
            system_prompt = (
                "You are a friendly customer service assistant. "
                "Rewrite the following message in a warm, natural tone, "
                "asking the user to provide more detail or rephrase their request. "
                "Keep it short (one or two sentences). Do not add bullet points or lists."
            )
            user_prompt = f"Original message: {text}\n\nRewrite:"
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_tokens=80,
                )
                result = (
                    completion.choices[0].message.content.strip()
                    if completion and completion.choices
                    else None
                )
                return result or text
            except Exception:
                return text

        formatted_options = (
            "\n" + "\n".join(f"  • {o.replace('_', ' ')}" for o in options)
        )

        # Wrap options in <readonly> XML tags — signals to the LLM that these
        # values are system-controlled and must not be reproduced or altered.
        options_xml = "<readonly>" + ", ".join(options) + "</readonly>"

        system_prompt = (
            "You are a friendly customer service assistant. "
            "Write ONE short, natural-sounding sentence asking the user to clarify "
            "which of the listed options applies to their request. "
            "The intent names are provided inside <readonly>...</readonly> tags. "
            "These are system-controlled values — do NOT reproduce, paraphrase, or "
            "include any of them in your response. They will be appended automatically. "
            "End your sentence with a colon (:) so the options can follow. "
            "Example: 'To help you better, could you tell me which of the following applies:'"
        )
        user_prompt = (
            f"The system needs to distinguish between these intents: {options_xml}.\n"
            "Write only the lead-in sentence (ending with a colon). "
            "Do NOT include any intent names from the <readonly> block in your response."
        )

        try:
            max_tokens = int(st.session_state.get("llm_max_tokens", int(os.getenv("MAX_TOKENS", "400"))))
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=60,  # just one sentence
            )
            intro = (
                completion.choices[0].message.content.strip()
                if completion and completion.choices
                else None
            )
            if intro:
                # Ensure the intro ends with a colon
                if not intro.rstrip().endswith(":"):
                    intro = intro.rstrip().rstrip("?.,") + ":"
                return intro + formatted_options
        except Exception:
            pass
        # Fallback: return original DS text if LLM fails
        return text

    elif response_type == "prediction":
        system_prompt = (
            "You rewrite final predictions to sound warm and natural. "
            "Preserve the intent label exactly as given. Keep it short "
            "and friendly, like a helpful assistant confirming they "
            "understood."
        )
    else:
        system_prompt = (
            "You rewrite explanations to sound warm, natural, and conversational, "
            "like a friendly assistant explaining their thought process. "
            "Use first-person narrative ('I was confused', 'you helped me understand'). "
            "Preserve all intent labels and candidate names exactly as given. "
            "Keep the conversational flow and story-telling style. "
            "Do not add technical jargon or numbers - keep it human and relatable."
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
        content = (
            completion.choices[0].message.content
            if completion and completion.choices
            else None
        )
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



# Module-level handle; populated by initialize_ds_systems() at startup.
_shared_embedder = None


def _load_ds_model_for_dataset(dataset_name: str):
    """Load DS model for a specific dataset. Returns None if not available."""
    configs = {
        'banking77': {
            'hierarchy':   'config/hierarchies/banking77_hierarchy.json',
            'intents':     'config/hierarchies/banking77_intents.json',
            'thresholds':  'results/banking77/workflow_demo/banking77_optimal_thresholds.json',
            'classifier':  'experiments/banking77/banking77_logistic_model.pkl',
        },
        'clinc150': {
            'hierarchy':   'config/hierarchies/clinc150_hierarchy.json',
            'intents':     'config/hierarchies/clinc150_intents.json',
            'thresholds':  'results/clinc150/workflow_demo/clinc150_optimal_thresholds.json',
            'classifier':  'experiments/clinc150/clinc150_logistic_model.pkl',
        },
    }
    cfg = configs.get(dataset_name)
    if cfg is None:
        return None

    # If classifier is missing locally, download it from Dropbox
    if not os.path.exists(cfg['classifier']):
        try:
            from src.utils.dropbox_saver import download_from_dropbox
            model_filename = os.path.basename(cfg['classifier'])
            dropbox_path = f"/ds_project_models/{model_filename}"
            print(f"⬇️ {dataset_name} model not found locally — downloading from Dropbox...")
            download_from_dropbox(dropbox_path, cfg['classifier'])
        except Exception as _e:
            print(f"⚠️ Dropbox download failed for {dataset_name}: {_e}")

    # Require at minimum the hierarchy and classifier
    missing = [p for p in [cfg['hierarchy'], cfg['classifier']] if not os.path.exists(p)]
    if missing:
        print(f"❌ [{dataset_name}] Cannot load — missing files: {missing}")
        return None

    try:
        hierarchy = load_hierarchy_from_json(cfg['hierarchy'])
        hierarchical_intents = load_hierarchical_intents_from_json(cfg['intents'])
        intent_embeddings = IntentEmbeddings(hierarchical_intents, embedder=_shared_embedder)
        classifier = IntentClassifier.from_pretrained(cfg['classifier'])
        custom_thresholds = None
        if os.path.exists(cfg['thresholds']):
            custom_thresholds = load_thresholds_from_json(cfg['thresholds'])
        print(f"✅ [{dataset_name}] Model loaded — {len(hierarchy)} intents, "
              f"thresholds={'custom' if custom_thresholds else 'default'}")
        return DSMassFunction(
            intent_embeddings=intent_embeddings.get_all_embeddings(),
            hierarchy=hierarchy,
            classifier=classifier,
            custom_thresholds=custom_thresholds,
            enable_belief_tracking=True,
            embedder=_shared_embedder,  # reuse one SentenceEmbedder for both datasets
        )
    except Exception as _exc:
        import traceback
        print(f"❌ [{dataset_name}] Failed to load DS model:\n{traceback.format_exc()}")
        return None


@st.cache_resource
def _get_shared_embedder():
    """Load the SentenceEmbedder once and share it across all dataset models."""
    print("Loading shared SentenceEmbedder (intfloat/e5-base)...")
    embedder = SentenceEmbedder(model_name='intfloat/e5-base')
    gc.collect()
    return embedder


@st.cache_resource
def _get_ds_model_cached(dataset_name: str):
    """Load a single DS model lazily, cached per dataset.

    Models are only loaded on first access — banking77 warms up at startup;
    clinc150 loads the first time a clinc150 query is served. This keeps
    peak memory low on Community Cloud.
    """
    global _shared_embedder
    if _shared_embedder is None:
        _shared_embedder = _get_shared_embedder()
    model = _load_ds_model_for_dataset(dataset_name)
    gc.collect()
    return model


def get_ds_system(dataset_name: str):
    """Return (ds_system, active_dataset, fallback_used) for the given dataset."""
    model = _get_ds_model_cached(dataset_name)
    if model is not None:
        return model, dataset_name, False
    # Fallback to banking77
    fallback = _get_ds_model_cached('banking77')
    return fallback, 'banking77', True

@st.cache_data
def load_study_queries():
    """Load the study query set, downloading from Dropbox if not cached locally.

    Directory is controlled by STUDY_SET_DIR env variable:
      unset / default → outputs/user_study/workflow_demo/  (original equal-split sets)
      study_v2        → outputs/user_study/study_v2/       (75/25 b77/clinc150 sets)

    The active set within the directory is controlled by STUDY_SET:
      small  → study_set_small.csv   (default)
      medium → study_set_medium.csv
      large  → study_set_large.csv
      full   → selected_queries_for_user_study.csv  (workflow_demo only)

    CSVs are stored in Dropbox and downloaded on first use.
    Dropbox path mirrors the local dir name: /ds_project_queries/<dir_basename>/
    """
    # Resolve STUDY_SET_DIR — check os.environ first, then st.secrets (Streamlit Cloud).
    _dir_default = 'outputs/user_study/workflow_demo'
    _base = (
        os.getenv('STUDY_SET_DIR')
        or (st.secrets.get('STUDY_SET_DIR') if hasattr(st, 'secrets') else None)
        or _dir_default
    ).rstrip('/')
    _dropbox_folder = '/ds_project_queries/' + os.path.basename(_base)
    # Resolve STUDY_SET — same priority: os.environ → st.secrets → 'small'
    _set_name = (
        os.getenv('STUDY_SET')
        or (st.secrets.get('STUDY_SET') if hasattr(st, 'secrets') else None)
        or 'small'
    ).strip().lower()

    # Map set name → (local path, Dropbox path)
    _candidates = {
        'small':  (f'{_base}/study_set_small.csv',  f'{_dropbox_folder}/study_set_small.csv'),
        'medium': (f'{_base}/study_set_medium.csv', f'{_dropbox_folder}/study_set_medium.csv'),
        'large':  (f'{_base}/study_set_large.csv',  f'{_dropbox_folder}/study_set_large.csv'),
        'full':   (f'{_base}/selected_queries_for_user_study.csv',
                   f'{_dropbox_folder}/selected_queries_for_user_study.csv'),
    }

    def _try_load(key):
        """Return DataFrame for key, downloading from Dropbox if needed. None if unavailable."""
        local_path, dropbox_path = _candidates[key]
        # Already cached locally?
        if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
            return pd.read_csv(local_path)
        # Try Dropbox
        try:
            from src.utils.dropbox_saver import download_from_dropbox
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            ok = download_from_dropbox(dropbox_path, local_path)
            if ok and os.path.exists(local_path):
                print(f"[load_study_queries] Downloaded '{key}' from Dropbox → {local_path}")
                return pd.read_csv(local_path)
        except Exception as _e:
            print(f"[load_study_queries] Dropbox download failed for '{key}': {_e}")
        return None

    # Try requested set first, then fall back through the others
    _order = [_set_name] + [k for k in ('small', 'medium', 'large', 'full') if k != _set_name]
    for _key in _order:
        df = _try_load(_key)
        if df is not None:
            if _key != _set_name:
                st.warning(
                    f"⚠️ STUDY_SET='{_set_name}' not available — loaded '{_key}' instead. "
                    f"Upload the CSV to Dropbox ({_dropbox_folder}/study_set_{_set_name}.csv) or "
                    f"run the EXPERIMENT SETS (B77-ONLY) cell in the workflow notebook."
                )
            print(f"[load_study_queries] Using set='{_key}' ({len(df)} queries)")
            return df

    st.error(
        f"No study query file found for STUDY_SET='{_set_name}' and Dropbox download failed. "
        "Upload the CSVs to Dropbox under /ds_project_queries/ or run the "
        "EXPERIMENT SETS cell in notebooks/system_workflow_demo.ipynb first."
    )
    st.stop()

def show_header():
    """Display header with HiCXAI styling"""
    query_number = st.session_state.get("current_query_index", 0) + 1

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

    st.markdown(f"""
    <div class="header-container">
        <h2 style="margin: 0; font-size: 2em;">Customer Service Assistant</h2>
        <p style="margin: 14px 0 8px 0; opacity: 0.95; line-height: 1.5; font-size: 1.08em;">
            This assistant helps with customer support requests and may ask follow-up questions when needed.
        </p>
        <p style="margin: 8px 0 0 0; opacity: 0.95; line-height: 1.5; font-size: 1.08em;">
            You are the customer seeking help for <strong style="color:#000000;">Customer Query #{query_number}</strong>, shown below. Please respond as you would in a real customer service conversation.
        </p>
        <p style="margin: 10px 0 0 0; opacity: 0.9; font-size: 0.98em; line-height: 1.45;">
            You may exchange up to 5 messages with the assistant. If the issue is still not resolved after 5 interactions, the outcome will be marked as <strong>Unknown</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

def format_bubble_text(text: str) -> str:
    """Format text for HTML chat bubbles with basic markdown support."""
    safe = html.escape(text)
    safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = safe.replace("\n", "<br>")
    return safe


def _get_tracker(ds_system):
    """Get belief tracker from DS system with fallback."""
    if hasattr(ds_system, 'get_belief_tracker'):
        tracker = ds_system.get_belief_tracker()
        if tracker is not None:
            return tracker
    return BeliefTracker()

def generate_belief_visualization(ds_system, title="Belief Progression"):
    """Generate real-time belief visualization for leaf intents only."""
    try:
        tracker = _get_tracker(ds_system)
        history = tracker.get_history()
        if not history:
            st.session_state['_viz_error'] = (
                f"generate_belief_visualization: tracker has no history. "
                f"tracker={tracker!r}, belief_tracker attr={getattr(ds_system, 'belief_tracker', 'MISSING')!r}"
            )
            return None

        # Extract leaf intents only (actual decision points)
        all_intents = set()
        for belief_dict, _ in history:
            all_intents.update(belief_dict.keys())

        hierarchy = ds_system.hierarchy if hasattr(ds_system, 'hierarchy') else {}
        leaf_intents = [
            intent for intent in all_intents
            if intent not in hierarchy or not hierarchy[intent]
        ]

        # Fallback: if no leaves found (hierarchy mismatch), use all intents
        if not leaf_intents:
            leaf_intents = list(all_intents)

        # Filter history to show only leaf intents
        filtered_history = [
            ({intent: belief_dict.get(intent, 0.0) for intent in leaf_intents}, label)
            for belief_dict, label in history
        ]

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            BeliefVisualizer.plot_top_intents_progression(
                belief_history=filtered_history,
                top_k=min(5, len(leaf_intents)),
                title=title,
                save_path=tmp.name,
                figsize=(10, 6)
            )
            tmp_path = tmp.name

        with open(tmp_path, "rb") as img_file:
            img_bytes = img_file.read()
        os.unlink(tmp_path)
        return img_bytes  # raw bytes — safe to store in session_state

    except Exception as e:
        import traceback as _tb
        st.session_state['_viz_error'] = f"Visualization error: {str(e)}\n{_tb.format_exc()}"
        return None

def process_query(query_text, ds_system, is_initial=True, previous_mass=None):
    """Process a single DS turn synchronously — mirrors stable-25Feb pattern.

    Returns (response_text, needs_clarification, combined_mass).
    combined_mass must be stored in st.session_state.current_mass and passed back
    as previous_mass on every follow-up turn so evidence accumulates correctly.
    """
    try:
        if is_initial:
            ds_system.conversation_history = []
            ds_system.user_response = None
            if hasattr(ds_system, 'clear_belief_history'):
                ds_system.clear_belief_history()

        current_mass = ds_system.compute_mass_function(query_text)

        if previous_mass is not None and not is_initial:
            combined_mass = ds_system.combine_mass_functions(previous_mass, current_mass)
        else:
            combined_mass = current_mass

        # Record belief state for explainability charts.
        # Mirror evaluate_from_leaves(depth): depth==0 → "Initial", depth==N → "Turn N".
        # conversation_history is unused in the Streamlit path (no blocking callback), so
        # we derive the label from how many snapshots are already stored in the tracker.
        belief = ds_system.compute_belief(combined_mass)
        if hasattr(ds_system, 'get_belief_tracker'):
            tracker = ds_system.get_belief_tracker()
            if tracker is not None:
                existing_records = len(tracker.get_history())
                turn_label = "Initial" if existing_records == 0 else f"Turn {existing_records}"
                tracker.record_belief(belief, turn_label)

        # Use get_clarification_step — exact mirror of notebook's evaluate_from_leaves():
        # Case 1a (single confident leaf)  → (None, intent, confidence) → predict
        # Case 1b (multiple confident)     → (question, None, 0.0)      → ask
        # Case 3  (no confident nodes)     → (question, None, 0.0)      → rephrase
        clarification_q, pred_intent_direct, confidence_direct = ds_system.get_clarification_step(combined_mass)

        if clarification_q is not None:
            # Needs clarification (Case 1b or Case 3)
            clarification = clarification_q
            if not clarification.endswith("?"):
                clarification += "?"
            clarification += " Or feel free to describe what you need in your own words."
            clarification = _humanize_response(clarification, response_type="clarification", context={})
            return clarification, True, combined_mass

        # Confident prediction (Case 1a) — use values already returned by get_clarification_step
        pred_intent = pred_intent_direct if pred_intent_direct else "unknown"
        confidence = confidence_direct if confidence_direct else 0.0
        st.session_state.last_prediction = pred_intent
        st.session_state.last_confidence = confidence

        tracker = ds_system.get_belief_tracker() if hasattr(ds_system, 'get_belief_tracker') else None
        n_turns = len(tracker.get_history()) if tracker else 0
        if n_turns >= 2:
            img = generate_belief_visualization(ds_system, "Belief Progression")
            st.session_state.last_belief_plot = img
            st.session_state.show_belief_chart = img is not None
        else:
            st.session_state.last_belief_plot = None
            st.session_state.show_belief_chart = False

        response = "Thank you for the information. I have now reviewed your request and made a prediction."
        return response, False, combined_mass

    except Exception as e:
        import traceback as _tb
        st.error(f"DS Error: {str(e)}")
        st.error(_tb.format_exc())
        return "I encountered an error. Could you rephrase?", True, None


def generate_confidence_explanation(ds_system):
    """Generate confidence explanation using BeliefVisualizer."""
    try:
        tracker = _get_tracker(ds_system)

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
            img_bytes = img_file.read()
        os.unlink(tmp_path)
        return img_bytes  # raw bytes — safe to store in session_state

    except Exception as e:
        import traceback as _tb
        st.session_state['_viz_error'] = f"Confidence visualization error: {str(e)}\n{_tb.format_exc()}"
        return None


def _init_session_defaults():
    """Initialize Streamlit session state defaults."""
    defaults = {
        'current_query_index': 0,
        'conversation_history': [],
        'awaiting_clarification': False,
        'current_mass': None,  # accumulated DS mass function across turns
        'session_results': [],
        'session_id': str(uuid.uuid4())[:8],
        'query_start_time': None,
        'result_saved': False,
        'feedback_submitted': False,
        'last_prediction': None,
        'last_confidence': None,
        'conversation_started': False,
        'query_resolved': False,
        'last_belief_plot': None,
        'show_belief_chart': False,
        'humanize_responses': True,
        'llm_model_name': os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        'llm_temperature': float(os.getenv("TEMPERATURE", "0.6")),
        'llm_max_tokens': int(os.getenv("MAX_TOKENS", "400")),
        'llm_warning_shown': False,
        'clarification_turns': 0,
        'feedback_stage': 'ranking',
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application"""

    # Both models load lazily on first use via _get_ds_model_cached (@st.cache_resource).
    # Nothing is pre-warmed at startup — keeps Community Cloud within memory limits.
    queries_df = load_study_queries()
    
    # Initialize data logger (for GitHub save)
    if 'data_logger_initialized' not in st.session_state:
        init_logger()
        st.session_state.data_logger_initialized = True
    
    # Initialize session state
    _init_session_defaults()
    
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
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Queries Completed", completed)
            with col2:
                st.metric("Avg Interactions", f"{avg_interactions:.1f}")
            with col3:
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
    
    # Get current query and lazily load the matching DS model
    current_query = queries_df.iloc[st.session_state.current_query_index]
    query_text = current_query['query']
    _query_dataset = str(current_query.get('dataset', 'banking77')) if 'dataset' in current_query.index else 'banking77'
    ds_system, _active_dataset, _fallback_used = get_ds_system(_query_dataset)
    if ds_system is None:
        st.error('❌ No DS model available. Check experiments/ and config/hierarchies/.')
        st.stop()

    # Progress indicator with stats
    progress = st.session_state.current_query_index / len(queries_df)
    st.progress(progress)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        remaining = len(queries_df) - st.session_state.current_query_index
        st.caption(f"Query {st.session_state.current_query_index + 1} of {len(queries_df)} ({remaining} remaining)")
    with col2:
        if _fallback_used:
            st.warning(
                f"⚠️ Model for **{_query_dataset}** not loaded — "
                f"using **{_active_dataset}** (wrong intents). "
                f"Run the workflow notebook for {_query_dataset} first."
            )
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
                if st.session_state.conversation_started:
                    pred = st.session_state.get('last_prediction', 'unknown')
                    save_result_to_session(current_query, ds_system, pred,
                                          pred == current_query.get('true_intent', ''))
                st.session_state.current_query_index = len(queries_df)
                st.rerun()


    
    # Auto-start query processing (no button click needed)
    if not st.session_state.conversation_started:
        st.session_state.query_start_time = datetime.datetime.now()
        # Synchronous per-turn process_query() — mirrors stable-25Feb pattern
        response, needs_clarification, current_mass = process_query(query_text, ds_system, is_initial=True)
        st.session_state.current_mass = current_mass
        st.session_state.conversation_history = [
            f"User: {query_text}",
            f"Assistant: {response}"
        ]
        st.session_state.awaiting_clarification = needs_clarification
        st.session_state.query_resolved = not needs_clarification
        st.session_state.conversation_started = True
        # NO rerun - continue rendering
    
    # Show query at top when conversation started
    st.markdown(f"""
    <div class="query-card">
        <strong>Customer Query #{st.session_state.current_query_index + 1}:</strong><br>
        {query_text}
    </div>
    """, unsafe_allow_html=True)
    
    # Always show the conversation (including first-pass confident predictions)
    if st.session_state.conversation_history:
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

    # Surface any hidden visualization errors (cleared each render)
    if st.session_state.get('_viz_error'):
        st.error(st.session_state.pop('_viz_error'))

    # Belief progression chart is shown inside the feedback stages (after user asks why)
    # Only show it during active clarification turns (not once resolved)
    if st.session_state.get('last_belief_plot') and not st.session_state.get('query_resolved', False):
        st.markdown("### Belief Progression")
        _img_bytes = st.session_state['last_belief_plot']
        st.image(BytesIO(_img_bytes), caption="Top 5 Intents Over Time", width=520)
        with st.expander("View large chart", expanded=False):
            st.image(BytesIO(_img_bytes), caption="Top 5 Intents Over Time (Large)", width=1100)

    

    # Chat interface at bottom
    st.markdown("---")

    # Show status when resolved - feedback form will appear below
    if st.session_state.query_resolved:
        predicted_intent = st.session_state.get('last_prediction', 'unknown')
        true_intent = current_query.get('true_intent', '')
        is_correct = (predicted_intent == true_intent)

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
                is_correct,
                ds_system
            )

            if feedback_submitted:
                # Move to next query
                st.session_state.current_query_index += 1
                # Reset conversation state
                st.session_state.conversation_started = False
                st.session_state.conversation_history = []
                st.session_state.awaiting_clarification = False
                st.session_state.current_mass = None
                st.session_state.query_resolved = False
                st.session_state.query_start_time = None
                st.session_state.result_saved = False
                st.session_state.feedback_submitted = False
                st.session_state.last_belief_plot = None
                st.session_state.clarification_turns = 0
                st.session_state.feedback_stage = 'ranking'  # Reset for next query
                gc.collect()
                st.rerun()
    elif st.session_state.awaiting_clarification:
        st.info("Please provide more information or type 'why' to understand my question.")
    
    # Chat input - for clarifications, explanations, and "why" questions
    if st.session_state.query_resolved:
        placeholder = "Type here (use the feedback form below to proceed)"
    else:
        placeholder = "Type your response or ask 'why?'"
    
    user_input = st.chat_input(placeholder, key=f"chat_{st.session_state.current_query_index}")
    
    if user_input:
        user_input_lower = user_input.lower().strip()
        
        # Handle "why" questions (allowed during clarification AND after resolution)
        if 'why' in user_input_lower or 'how did you' in user_input_lower or 'explain' in user_input_lower:
            # During active clarification: explain inline as before
            if st.session_state.awaiting_clarification:
                st.session_state.conversation_history.append(f"User: {user_input}")
                if 'data_logger' in st.session_state and st.session_state.data_logger:
                    st.session_state.data_logger.log_why_question()
                explanation, belief_plot, _ = get_ds_explanation(ds_system, "clarification")
                explanation = _humanize_response(explanation, response_type="explanation",
                                                  context={"explanation_type": "clarification"})
                st.session_state.conversation_history.append(f"Assistant: {explanation}")
                st.session_state.last_belief_plot = belief_plot
                st.rerun()
            else:
                # Query resolved — explanation is handled in feedback stage 3
                st.info("Use the 'Why was this predicted?' button in the feedback section below.")
                st.rerun()

        # When resolved, all non-why input is ignored (feedback form handles advancing)
        elif st.session_state.query_resolved:
            st.info("Please use the feedback form below to continue.")
            st.rerun()
        
        # Handle clarification response (during active clarification)
        elif st.session_state.awaiting_clarification:
            st.session_state.conversation_history.append(f"User: {user_input}")
            st.session_state.clarification_turns += 1

            # Mirror notebook maximum_depth=5: give up after 5 clarification turns
            if st.session_state.clarification_turns >= 5:
                unresolved_msg = (
                    "I wasn't able to determine your intent after several attempts. "
                    "Query not resolved."
                )
                st.session_state.last_prediction = "unknown"
                st.session_state.last_confidence = 0.0
                st.session_state.conversation_history.append(f"Assistant: {unresolved_msg}")
                st.session_state.awaiting_clarification = False
                st.session_state.query_resolved = True
            else:
                # Synchronous per-turn: combine user answer with accumulated mass
                response, needs_clarification, current_mass = process_query(
                    user_input, ds_system,
                    is_initial=False,
                    previous_mass=st.session_state.current_mass
                )
                st.session_state.current_mass = current_mass
                st.session_state.conversation_history.append(f"Assistant: {response}")
                st.session_state.awaiting_clarification = needs_clarification
                st.session_state.query_resolved = not needs_clarification
            st.rerun()

def _build_ranked_options(query_index, predicted_intent, true_intent, ds_system):
    """Build the 4 intent options from DS belief ranking, shuffled once per query.

    Case A (predicted == oracle): top 4 by belief score.
    Case B (predicted != oracle):  predicted + oracle + next 2 by belief score.
    Returns a stable shuffled list of 4 intent strings (no Neither — appended at display time).
    """
    _key = f"ranked_options_{query_index}"
    if _key in st.session_state:
        return st.session_state[_key]

    # Derive full belief ranking from the accumulated mass function
    ranked = []
    combined_mass = st.session_state.get('current_mass')
    if combined_mass is not None and ds_system is not None:
        try:
            belief = ds_system.compute_belief(combined_mass)
            leaf_beliefs = [
                (intent, score) for intent, score in belief.items()
                if ds_system.is_leaf(intent)
            ]
            leaf_beliefs.sort(key=lambda x: x[1], reverse=True)
            ranked = [intent for intent, _ in leaf_beliefs]
        except Exception:
            pass

    # Fallback: if belief ranking unavailable, use predicted + oracle only
    if not ranked:
        ranked = [predicted_intent] if predicted_intent != 'unknown' else []

    if predicted_intent != 'unknown' and predicted_intent == true_intent:
        # Case A: top 4 from belief ranking
        options = [r for r in ranked if r][:4]
    else:
        # Case B: predicted + oracle, then fill to 4 from remaining ranked
        fixed = []
        if predicted_intent != 'unknown':
            fixed.append(predicted_intent)
        if true_intent not in ('unknown', '') and true_intent != predicted_intent:
            fixed.append(true_intent)
        extras = [r for r in ranked if r not in fixed]
        options = (fixed + extras)[:4]

    # Pad with intent names from hierarchy if we still have fewer than 4
    if len(options) < 4 and ds_system is not None:
        try:
            leaves = [i for i in ds_system.hierarchy if ds_system.is_leaf(i) and i not in options]
            options += leaves[:4 - len(options)]
        except Exception:
            pass

    # Shuffle once per query for unbiased display
    shuffled = options[:]
    random.shuffle(shuffled)
    st.session_state[_key] = shuffled
    return shuffled


def collect_query_feedback(query_index, query_text, predicted_intent, is_correct, ds_system):
    """Collect per-query feedback in 3 sequential stages.

    Stage 1 — ranking:             User ranks 4 intents (blind — no prediction revealed)
    Stage 2 — reveal_prediction:   System prediction shown; user can ask why
    Stage 3 — explanation_requested: Explanation + belief chart + ratings form → Submit & Continue
    """
    queries_df = load_study_queries()
    true_intent = queries_df.iloc[query_index]['true_intent'] if query_index < len(queries_df) else 'unknown'

    stage = st.session_state.get('feedback_stage', 'ranking')

    # ── STAGE 1: RANKING ────────────────────────────────────────────────────
    if stage == 'ranking':
        st.markdown("---")
        st.info(
            "This conversation has now ended. "
            "The assistant has made a prediction about your intended customer issue. "
            "Please rank the intent options below based on how closely they match what you meant."
        )

        st.markdown("### Rank the intent options")
        st.markdown(
            "Rank the four options from **best match** to **least match** "
            "based on your intended meaning in the conversation."
        )
        st.caption("Please base your ranking on what you meant, not on what you think the system selected.")

        intent_options = _build_ranked_options(query_index, predicted_intent, true_intent, ds_system)
        display_options = [o.replace('_', ' ') for o in intent_options]
        neither_label = "None of the above / Something else"

        rank_labels = [
            "**Rank 1 — Best match to your intent:**",
            "**Rank 2:**",
            "**Rank 3:**",
            "**Rank 4 — Least match:**",
        ]

        def _make_rank_on_change(rank_idx):
            """If the newly chosen value already appears in a later rank, clear that later rank."""
            def _cb():
                new_val = st.session_state.get(f"rank{rank_idx + 1}_{query_index}")
                if new_val is None:
                    return
                for j in range(rank_idx + 1, 4):
                    if st.session_state.get(f"rank{j + 1}_{query_index}") == new_val:
                        st.session_state[f"rank{j + 1}_{query_index}"] = None
            return _cb

        for i in range(4):
            # Options for this rank = full list minus what is already chosen in ranks above
            used = {
                st.session_state.get(f"rank{j + 1}_{query_index}")
                for j in range(i)
                if st.session_state.get(f"rank{j + 1}_{query_index}") is not None
            }
            available = [o for o in (display_options + [neither_label]) if o not in used]
            opts = [None] + available

            # If the stored value for this rank was filtered out by a higher rank, reset it
            cur = st.session_state.get(f"rank{i + 1}_{query_index}")
            if cur not in opts:
                st.session_state[f"rank{i + 1}_{query_index}"] = None

            st.markdown(rank_labels[i])
            st.selectbox(
                f"Rank {i + 1}",
                options=opts,
                format_func=lambda x: "Select…" if x is None else x,
                key=f"rank{i + 1}_{query_index}",
                on_change=_make_rank_on_change(i),
                label_visibility="collapsed",
            )

        sels = [st.session_state.get(f"rank{i + 1}_{query_index}") for i in range(4)]
        all_filled = all(s is not None for s in sels)

        if st.button(
            "Submit Ranking", type="primary", use_container_width=True,
            key=f"submit_ranking_{query_index}",
            disabled=not all_filled,
        ):
            st.session_state[f'user_ranking_{query_index}'] = list(sels)
            st.session_state.feedback_stage = 'reveal_prediction'
            st.rerun()
        return False

    # ── STAGE 2: REVEAL PREDICTION ──────────────────────────────────────────
    if stage == 'reveal_prediction':
        st.markdown("---")
        st.markdown("### System prediction")
        st.markdown("Thank you. Based on the conversation, the system predicted the following intent:")
        st.markdown(
            f"**Predicted intent:** `{predicted_intent.replace('_', ' ')}`"
        )
        st.markdown("---")
        st.markdown("You can now ask why this intent was predicted.")

        col_btn, col_txt = st.columns([1, 2])
        with col_btn:
            if st.button("Why was this predicted?", type="primary", key=f"why_btn_{query_index}"):
                st.session_state.feedback_stage = 'explanation_requested'
                st.rerun()
        with col_txt:
            why_text = st.text_input(
                "Or type a question about the prediction",
                placeholder="e.g. Why was this intent chosen?",
                key=f"why_text_{query_index}",
                label_visibility="collapsed"
            )
            if why_text:
                st.session_state[f'why_question_{query_index}'] = why_text
                st.session_state.feedback_stage = 'explanation_requested'
                st.rerun()
        return False

    # ── STAGE 3: EXPLANATION + RATINGS ──────────────────────────────────────
    if stage == 'explanation_requested':
        st.markdown("---")
        st.markdown("### Explanation of the prediction")

        # Generate and show explanation
        explanation, belief_plot, _ = get_ds_explanation(ds_system, "decision")
        explanation = _humanize_response(
            explanation,
            response_type="explanation",
            context={"explanation_type": "decision"}
        )
        st.markdown(explanation)

        # Belief progression chart
        if belief_plot is None:
            belief_plot = st.session_state.get('last_belief_plot')
        if belief_plot is None:
            belief_plot = generate_belief_visualization(ds_system, "Belief Progression")
        if belief_plot:
            st.markdown("The chart below shows how the system's confidence changed during the interaction.")
            st.image(BytesIO(belief_plot), caption="Belief Progression", width=520)
            with st.expander("View large chart", expanded=False):
                st.image(BytesIO(belief_plot), caption="Belief Progression (Large)", width=1100)

        st.markdown("---")
        st.markdown("##### Rate the interaction:")

        with st.form(f"ratings_form_{query_index}", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                clarity = st.select_slider(
                    "Was the assistant's response clear?",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    format_func=lambda x: "⭐" * x,
                    key=f"clarity_{query_index}",
                    help="1=Very unclear, 5=Very clear"
                )
            with col2:
                confidence_rating = st.select_slider(
                    "How confident are you in the result?",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    format_func=lambda x: "⭐" * x,
                    key=f"confidence_rating_{query_index}",
                    help="1=Not confident, 5=Very confident"
                )
            comment = st.text_input(
                "Any comments or suggestions? (optional)",
                placeholder="e.g. 'Too many clarifying questions' or 'The response was spot on'",
                key=f"comment_{query_index}"
            )
            submitted = st.form_submit_button("Submit & Continue", type="primary", use_container_width=True)

            if submitted:
                ranks = st.session_state.get(f'user_ranking_{query_index}', [None, None, None, None])
                # top-ranked intent (rank 1) is the user's best match
                user_top_intent_display = ranks[0] if ranks else None
                # convert display label back to underscore form for matching
                user_top_intent = user_top_intent_display.replace(' ', '_') if user_top_intent_display else None

                if 'session_results' in st.session_state and st.session_state.session_results:
                    last = st.session_state.session_results[-1]
                    last['user_ranking'] = ranks
                    last['user_validated_intent'] = user_top_intent
                    last['user_agrees_with_system'] = (user_top_intent == predicted_intent)
                    last['user_agrees_with_oracle'] = (user_top_intent == true_intent)
                    last['feedback_clarity'] = clarity
                    last['feedback_confidence'] = confidence_rating
                    last['feedback_comment'] = comment
                    last['feedback_submitted'] = True
                    if 'data_logger' in st.session_state and st.session_state.data_logger:
                        st.session_state.data_logger.log_query_result(last)

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
        
        result = _create_result_dict(
            query_row,
            predicted_intent,
            is_correct,
            confidence,
            clarification_turns,
            interaction_time,
            end_time
        )
        
        st.session_state.session_results.append(result)
        
    except Exception as e:
        st.error(f"Error saving query result: {str(e)}")

def _create_result_dict(
    query_row,
    predicted_intent: str,
    is_correct: bool,
    confidence: float,
    clarification_turns: int,
    interaction_time: float,
    end_time: datetime.datetime
) -> Dict:
    """Create standardized result dictionary."""
    return {
        'session_id': st.session_state.session_id,
        'query_index': st.session_state.current_query_index,
        'query_text': query_row['query'],
        'true_intent': query_row['true_intent'],
        'predicted_intent': predicted_intent,
        'confidence': confidence,
        'num_clarification_turns': clarification_turns,
        'is_correct': is_correct,
        'interaction_time_seconds': interaction_time,
        'conversation_transcript': '\n'.join(st.session_state.conversation_history)[-3000:],  # cap to save memory
        'timestamp': end_time.isoformat(),
        'llm_predicted_intent': query_row.get('predicted_intent', ''),
        'llm_num_interactions': query_row.get('num_interactions', 0),
        'llm_confidence': query_row.get('confidence', 0.0),
        'llm_was_correct': query_row.get('is_correct', False),
        'user_validated_intent': None,
        'user_ranking': None,
        'user_agrees_with_system': None,
        'user_agrees_with_oracle': None,
        'feedback_clarity': None,
        'feedback_confidence': None,
        'feedback_comment': '',
        'feedback_submitted': False
    }

def get_ds_explanation(ds_system, explanation_type):
    """Get real explainability with visual components"""
    try:
        if not hasattr(ds_system, 'get_belief_tracker') or not ds_system.get_belief_tracker():
            return "Belief tracking not available. Using Dempster-Shafer theory for reasoning.", None, None
            
        tracker = ds_system.get_belief_tracker()
        final_belief = tracker.get_final_belief()
        history = tracker.get_history()
        
        if explanation_type == "clarification":
            # Get actual reasoning for clarification (user-friendly, no technical jargon)
            text_explanation = "I asked for clarification because I noticed several similar options and wanted a bit more detail. Here are the top candidates: "
            
            if final_belief:
                # Show top candidates WITHOUT technical details (no scores/thresholds)
                sorted_beliefs = sorted(final_belief.items(), key=lambda x: x[1], reverse=True)[:3]
                candidate_names = [intent for intent, _ in sorted_beliefs]
                text_explanation += ", ".join(candidate_names) + "."
                
                # Reuse already-generated image (cached when prediction was made)
                belief_plot = st.session_state.get('last_belief_plot')
                if belief_plot is None:
                    belief_plot = generate_belief_visualization(ds_system, "Why I Asked for Clarification")
                confidence_plot = generate_confidence_explanation(ds_system)
                
                return text_explanation, belief_plot, confidence_plot
            
            return text_explanation, None, None
        
        elif explanation_type == "decision":
            # Get actual reasoning for decision (narrative, user-friendly)
            text_explanation = ""
            
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
                
                # Check if clarification ACTUALLY happened using the belief chart flag.
                # show_belief_chart is set True inside process_query() only when the belief
                # tracker records >= 2 turns (i.e. the DS system went through multiple rounds).
                # Counting user messages is unreliable because "why" questions also add user turns.
                had_clarification = st.session_state.get('show_belief_chart', False)
                
                if had_clarification:
                    # Clarification happened - tell the story of the conversation
                    # Get the candidates that were unclear initially
                    initial_belief = history[0][0] if history else {}
                    sorted_initial = sorted(initial_belief.items(), key=lambda x: x[1], reverse=True)[:3]
                    candidate_names = [intent for intent, _ in sorted_initial]
                    candidates_text = ", ".join(candidate_names)

                    text_explanation = (
                        f"I was a bit unsure at first — your query could have matched a few things: {candidates_text}. "
                        f"I asked you to clarify, and your reply helped me understand that you want: **{top_intent.replace('_', ' ')}**. "
                        f"You can see in the chart below how my confidence built up as we talked."
                    )
                else:
                    # No clarification - immediate decision, no graph to show
                    text_explanation = (
                        f"Your query was clear enough for me to see straight away that you want: **{top_intent.replace('_', ' ')}**. "
                        f"I felt confident enough to decide right away without needing any clarification."
                    )
                
                # Reuse already-generated image (cached when prediction was made)
                belief_plot = st.session_state.get('last_belief_plot')
                if belief_plot is None:
                    belief_plot = generate_belief_visualization(ds_system, "Decision Reasoning")
                confidence_plot = generate_confidence_explanation(ds_system)
                
                return text_explanation, belief_plot, confidence_plot
            
            fallback_intent = st.session_state.get('last_prediction')
            fallback_conf = st.session_state.get('last_confidence')
            if fallback_intent is not None and fallback_conf is not None:
                text_explanation = (
                    f"Based on your query, I identified your intent as **{fallback_intent.replace('_', ' ')}** "
                    f"with {fallback_conf:.0%} confidence."
                )
            else:
                text_explanation = "I processed your query using Dempster-Shafer reasoning to identify your intent."
            return text_explanation, None, None
    
    except Exception as e:
        return f"Error generating explanation: {str(e)}", None, None
    
    return "Using Dempster-Shafer theory for banking intent reasoning.", None, None

if __name__ == "__main__":
    main()