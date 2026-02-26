"""
Test the 'stolen wallet' query through DS to understand what cases it triggers.
Run with: /home/kudzai/miniconda3/envs/dsproject/bin/python test_stolen_wallet.py
"""
import sys, os
sys.path.insert(0, '.')

from config.hierarchy_loader import load_hierarchy_from_json, load_hierarchical_intents_from_json
from config.threshold_loader import load_thresholds_from_json
from src.models.ds_mass_function import DSMassFunction
from src.models.embeddings import IntentEmbeddings, SentenceEmbedder
from src.models.classifier import IntentClassifier

print("=== Loading models ===")
embedder = SentenceEmbedder(model_name='intfloat/e5-base')
hierarchy   = load_hierarchy_from_json('config/hierarchies/banking77_hierarchy.json')
hier_intents = load_hierarchical_intents_from_json('config/hierarchies/banking77_intents.json')
classifier   = IntentClassifier.from_pretrained('experiments/banking77/banking77_logistic_model.pkl')
thresholds   = load_thresholds_from_json('results/banking77/workflow_demo/banking77_optimal_thresholds.json')
intent_emb   = IntentEmbeddings(hier_intents, embedder=embedder)

ds = DSMassFunction(
    intent_embeddings=intent_emb.get_all_embeddings(),
    hierarchy=hierarchy,
    classifier=classifier,
    custom_thresholds=thresholds,
    enable_belief_tracking=True,
    embedder=embedder,
)
print("Models loaded OK.\n")

query = "Someone stole my wallet earlier today, not sure exactly when, probably on Piccadilly circus. Can you check if there were any attempts to use the card and obviously block it?"
print(f"Query: {query}\n")

# --- Step 1: raw mass ---
mass = ds.compute_mass_function(query)
print("--- Top 8 mass values ---")
for k, v in sorted(mass.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"  {k}: {v:.4f}")

# --- Step 2: belief ---
belief = ds.compute_belief(mass)
print("\n--- Top 8 belief values vs thresholds ---")
for k, v in sorted(belief.items(), key=lambda x: x[1], reverse=True)[:8]:
    t = thresholds.get(k, None)
    leaf = not bool(hierarchy.get(k))
    confident = (v >= t) if t is not None else False
    print(f"  {k}: belief={v:.4f}  threshold={t}  leaf={leaf}  confident={confident}")

# --- Step 3: evaluate_hierarchy on leaf nodes ---
leaf_nodes = [n for n in hierarchy if not hierarchy.get(n)]
eval_result = ds.evaluate_hierarchy(leaf_nodes, mass)
print(f"\n--- evaluate_hierarchy result ---")
print(f"  type: {type(eval_result)}")
if isinstance(eval_result, (list, tuple)):
    for i, v in enumerate(eval_result):
        if isinstance(v, dict):
            print(f"  [{i}] dict with {len(v)} keys, sample: {dict(list(v.items())[:3])}")
        else:
            print(f"  [{i}]: {repr(v)[:200]}")
else:
    print(f"  result: {repr(eval_result)[:400]}")

# --- Step 4: walk through full evaluate_with_clarifications to see turns ---
print("\n=== evaluate_with_clarifications simulation ===")
print("(Callback auto-picks first option or gives a short rephrased answer)\n")

turn_num = [0]
answers_given = []

def auto_callback(conversation_history_str: str, question: str) -> str:
    turn_num[0] += 1
    # Extract options from the question (they appear as a python list in parens)
    import re
    opts = re.findall(r"\['(.+?)'\]", question) or re.findall(r"\((\[.+?\])\)", question)
    # simpler: just look for [...] in the question
    options_match = re.search(r"\(\[(.+?)\]\)", question)
    options = []
    if options_match:
        options = [o.strip().strip("'") for o in options_match.group(1).split(',')]
    
    print(f"[Turn {turn_num[0]}] DS asks: {repr(question[:120])}")
    print(f"  detected options: {options}")
    
    if options:
        answer = options[0]
    else:
        # CASE 3 rephrase — give clear rephrased answer
        answer = "I need to block my stolen card and check for fraudulent transactions"
    
    answers_given.append((question[:80], options, answer))
    print(f"  => auto-answer: {repr(answer)}\n")
    return answer

ds.customer_agent_callback = auto_callback

try:
    mass_init = ds.compute_mass_function(query)
    prediction = ds.evaluate_with_clarifications(mass_init)
    print(f"Final prediction: {prediction}")
    print(f"Total turns asked: {turn_num[0]}")
    tracker = ds.get_belief_tracker()
    if tracker:
        history = tracker.get_history()
        print(f"\nBelief history ({len(history)} entries):")
        for entry in history:
            label = entry[1] if isinstance(entry, tuple) and len(entry) > 1 else "?"
            beliefs_dict = entry[0] if isinstance(entry, tuple) else entry
            top = sorted(beliefs_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  [{label}] top3: {[(k, round(v,3)) for k,v in top]}")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
