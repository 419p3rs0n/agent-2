import os
import json
import re
import ast
import logging
import openai

from validators import verify_structural_accuracy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# Environment Variables
# ==========================

FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

FIREWORKS_BASE_URL = os.getenv(
    "FIREWORKS_BASE_URL",
    "https://api.fireworks.ai/inference/v1"
)

ALLOWED_MODELS = [
    m.strip()
    for m in os.getenv("ALLOWED_MODELS", "").split(",")
    if m.strip()
]

REMOTE_TARGET_MODEL = (
    ALLOWED_MODELS[0]
    if ALLOWED_MODELS
    else "accounts/fireworks/models/llama-v3-8b-instruct"
)

# ==========================
# Clients
# ==========================

local_client = openai.Client(
    base_url="http://ollama:11434/v1",
    api_key="ollama"
)

remote_client = openai.Client(
    base_url=FIREWORKS_BASE_URL,
    api_key=FIREWORKS_API_KEY
)


# ==========================
# Deterministic Solver
# ==========================

def run_deterministic_pass(prompt, category):

    if "math" in category.lower():

        match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", prompt)

        if match:

            try:
                return str(ast.literal_eval(match.group(0)))
            except Exception:
                pass

    return None


# ==========================
# Main Pipeline
# ==========================

def execute_routing_pipeline(task):

    prompt = task.get("prompt", "")
    category = task.get("category", "general")
    task_id = task.get("id")

    # ---------------------------------
    # Tier 1
    # ---------------------------------

    answer = run_deterministic_pass(prompt, category)

    if answer:

        logger.info(f"Task {task_id}: solved deterministically.")

        return {
            "id": task_id,
            "answer": answer
        }

    # ---------------------------------
    # Tier 2 - Ollama
    # ---------------------------------

    try:

        response = local_client.chat.completions.create(

            model="llama3.2",

            messages=[
                {
                    "role": "system",
                    "content":
                    "Answer the user's question directly. "
                    "Do not explain your reasoning unless asked."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0

        )

        answer = response.choices[0].message.content.strip()

        logger.info(f"Task {task_id}: answered using Ollama.")

        # Return immediately if Ollama produced anything
        if answer:

            return {
                "id": task_id,
                "answer": answer
            }

    except Exception as e:

        logger.warning(f"Ollama failed: {e}")

    # ---------------------------------
    # Tier 3 - Fireworks (ONLY if Ollama failed)
    # ---------------------------------

    if FIREWORKS_API_KEY:

        try:

            response = remote_client.chat.completions.create(

                model=REMOTE_TARGET_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0,
                max_tokens=100

            )

            logger.info(f"Task {task_id}: answered using Fireworks.")

            return {
                "id": task_id,
                "answer": response.choices[0].message.content
            }

        except Exception as e:

            logger.error(f"Fireworks failed: {e}")

    return {
        "id": task_id,
        "answer": "[Unable to generate response]"
    }


# ==========================
# Main
# ==========================

def main():

    if os.path.exists("/input/tasks.json"):

        input_path = "/input/tasks.json"
        output_path = "/output/results.json"

    else:

        input_path = "input/tasks.json"
        output_path = "output/results.json"

    if not os.path.exists(input_path):

        print("tasks.json not found.")

        return

    with open(input_path, "r") as f:

        tasks = json.load(f)

    results = []

    for task in tasks:

        results.append(
            execute_routing_pipeline(task)
        )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(output_path, "w") as f:

        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()