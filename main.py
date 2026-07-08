import os
import json
import re
import ast
import logging
import openai
from validators import verify_structural_accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse official platform runtime environmental variables
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
ALLOWED_MODELS = [m.strip() for m in os.getenv("ALLOWED_MODELS", "").split(",") if m.strip()]

# Validate API configuration
if not FIREWORKS_API_KEY:
    logger.warning("FIREWORKS_API_KEY not set; remote fallback will fail")

REMOTE_TARGET_MODEL = ALLOWED_MODELS[0] if ALLOWED_MODELS else "accounts/fireworks/models/llama-v3-8b-instruct"

# Establish Endpoint SDK Handlers
local_client = openai.Client(base_url="http://localhost:11434/v1", api_key="ollama")
remote_client = openai.Client(base_url=FIREWORKS_BASE_URL, api_key=FIREWORKS_API_KEY)

def run_deterministic_pass(prompt, category):
    """Tier 1: Regex solver to bypass LLMs entirely on basic arithmetic calculations (0 tokens)."""
    if "math" in category.lower():
        math_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', prompt)
        if math_match:
            try:
                # Safe evaluation using ast instead of eval
                return str(ast.literal_eval(math_match.group(0)))
            except (ValueError, SyntaxError):
                return None
    return None

def execute_routing_pipeline(task):
    prompt = task.get("prompt", "")
    category = task.get("category", "general")
    task_id = task.get("id")

    # Tier 1: Deterministic evaluation pass
    static_solution = run_deterministic_pass(prompt, category)
    if static_solution:
        return {"id": task_id, "answer": static_solution}

    # Tier 2: Free Local Generation with Gemma 2
    local_draft = "[Local Generation Failed]"
    try:
        local_response = local_client.chat.completions.create(
            model="gemma2:7b",
            messages=[
                {"role": "system", "content": "[SYSTEM: STRICT LOGIC ENGINE]\nExecute task directly. Zero filler. If schema required, output ONLY schema."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0  # Force maximum determinism
        )
        local_draft = local_response.choices[0].message.content

        # Tier 3: Core Validation Gatekeeper
        if verify_structural_accuracy(category, local_draft):
            return {"id": task_id, "answer": local_draft}
    except Exception as e:
        logger.warning(f"Local generation failed for task {task_id}: {e}")
        local_draft = "[Local Generation Failed]"

    # Tier 4: Low-Token Specular Fallback (Fireworks API)
    # Treating the remote model purely as an editor to patch errors keeps token usage tiny
    if not FIREWORKS_API_KEY:
        logger.error(f"Remote fallback skipped for task {task_id}: FIREWORKS_API_KEY not set")
        return {"id": task_id, "answer": local_draft}
    
    try:
        remote_response = remote_client.chat.completions.create(
            model=REMOTE_TARGET_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Fix/Patch: {prompt}\nFlawed Draft: {local_draft}"}
            ],
            max_tokens=35,  # Strict generation ceiling to safeguard token metrics
            temperature=0.0
        )
        return {"id": task_id, "answer": remote_response.choices[0].message.content}
    except Exception as e:
        logger.error(f"Remote API failed for task {task_id}: {e}")
        return {"id": task_id, "answer": local_draft}

def main():
    # Automatically switch between Hackathon Server paths and local Windows paths
    if os.path.exists("/input/tasks.json"):
        input_path = "/input/tasks.json"
        output_path = "/output/results.json"
    else:
        input_path = "input/tasks.json"
        output_path = "output/results.json"

    if not os.path.exists(input_path):
        print(f"Error: Local test file missing! Please create an 'input' folder in your sidebar and add a 'tasks.json' inside it.")
        return

    with open(input_path, "r") as f:
        tasks = json.load(f)

    results = []
    for task in tasks:
        output_node = execute_routing_pipeline(task)
        results.append(output_node)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Success! Local evaluation processed. Results saved to: {output_path}")

# Explicit script execution trigger
if __name__ == "__main__":
    main()
