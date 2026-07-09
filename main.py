import os
import json
import re
import ast
import logging
import openai

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
# Model Capability Profiles
# ==========================

MODELS = {

    "ollama": {
        "capability": 70,
        "cost": 0,
        "model": "llama3.2"
    },

    "fireworks": {
        "capability": 95,
        "cost": 1,
        "model": REMOTE_TARGET_MODEL
    }
}


# ==========================
# Deterministic Solver
# ==========================

def run_deterministic_pass(prompt, category):

    if "math" in category.lower():

        match = re.search(
            r"(\d+)\s*([\+\-\*\/])\s*(\d+)",
            prompt
        )

        if match:

            try:
                return str(
                    ast.literal_eval(match.group(0))
                )

            except Exception:
                pass

    return None



# ==========================
# Intelligence Layer
# ==========================

def estimate_tokens(prompt):

    return int(len(prompt.split()) * 1.3)



def estimate_complexity(prompt):

    score = 20

    text = prompt.lower()

    words = len(prompt.split())


    if words > 50:
        score += 15

    if words > 100:
        score += 20


    keywords = {

        "explain": 10,
        "compare": 15,
        "analyze": 25,
        "design": 25,
        "implement": 20,
        "algorithm": 20,
        "architecture": 25,
        "security": 15,
        "cybersecurity": 15,
        "code": 20,
        "python": 15,
        "optimize": 20,
        "step by step": 15
    }


    for word, value in keywords.items():

        if word in text:
            score += value


    return min(score,100)



def choose_route(prompt):

    complexity = estimate_complexity(prompt)


    # Local model is enough
    if complexity <= MODELS["ollama"]["capability"]:

        return {

            "route": "ollama",
            "model": "llama3.2",
            "confidence": round(
                1 - (complexity / 200),
                2
            ),
            "reason":
            f"Complexity {complexity} fits local model",
            "estimated_tokens":
            estimate_tokens(prompt)

        }


    # Harder tasks
    return {

        "route": "fireworks",
        "model": REMOTE_TARGET_MODEL,
        "confidence": 0.95,
        "reason":
        f"Complexity {complexity} requires stronger model",
        "estimated_tokens":
        estimate_tokens(prompt)

    }



# ==========================
# Model Calls
# ==========================

def ask_ollama(prompt):

    response = local_client.chat.completions.create(

        model="llama3.2",

        messages=[

            {
                "role": "system",
                "content":
                "Answer directly and accurately."
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0

    )

    return response.choices[0].message.content.strip()



def ask_fireworks(prompt):

    response = remote_client.chat.completions.create(

        model=REMOTE_TARGET_MODEL,

        messages=[

            {
                "role": "system",
                "content":
                "Answer directly and accurately."
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0,

        max_tokens=200

    )

    return response.choices[0].message.content.strip()



# ==========================
# Routing Pipeline
# ==========================

def execute_routing_pipeline(task):

    prompt = task.get(
        "prompt",
        ""
    )

    category = task.get(
        "category",
        "general"
    )

    task_id = task.get(
        "id"
    )


    # --------------------------
    # Tier 0: Deterministic
    # --------------------------

    answer = run_deterministic_pass(
        prompt,
        category
    )


    if answer:

        logger.info(
            f"Task {task_id}: deterministic"
        )

        return {

            "id": task_id,
            "answer": answer,
            "route": "deterministic",
            "confidence": 1.0,
            "reason": "Solved locally without LLM",
            "estimated_tokens": 0

        }



    decision = choose_route(prompt)


    logger.info(
        f"Task {task_id}: {decision['route']} - {decision['reason']}"
    )


    # --------------------------
    # Ollama Route
    # --------------------------

    if decision["route"] == "ollama":

        try:

            answer = ask_ollama(prompt)


            if answer:

                return {

                    "id": task_id,
                    "answer": answer,
                    **decision

                }


        except Exception as e:

            logger.warning(
                f"Ollama failed: {e}"
            )



    # --------------------------
    # Fireworks Route
    # --------------------------

    if FIREWORKS_API_KEY:

        try:

            answer = ask_fireworks(prompt)


            return {

                "id": task_id,
                "answer": answer,
                **decision

            }


        except Exception as e:

            logger.error(
                f"Fireworks failed: {e}"
            )


    return {

        "id": task_id,
        "answer":
        "[Unable to generate response]",
        "route":
        "failed"

    }



# ==========================
# Main
# ==========================

def main():

    if os.path.exists("/input/tasks.json"):

        input_path="/input/tasks.json"
        output_path="/output/results.json"

    else:

        input_path="input/tasks.json"
        output_path="output/results.json"



    if not os.path.exists(input_path):

        print(
            "tasks.json not found"
        )

        return



    with open(input_path,"r") as f:

        tasks=json.load(f)



    results=[]


    for task in tasks:

        results.append(
            execute_routing_pipeline(task)
        )



    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )


    with open(output_path,"w") as f:

        json.dump(
            results,
            f,
            indent=4
        )


    print(
        f"Results saved to {output_path}"
    )



if __name__=="__main__":

    main()