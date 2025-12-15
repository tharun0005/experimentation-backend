import time
import os
import numpy as np
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from clearml import Task, OutputModel
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import json
from pathlib import Path
import yaml
import sys

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['TQDM_DISABLE'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
BACKEND_HOST = os.getenv('BACKEND_HOST', '0.0.0.0')
BACKEND_PORT = int(os.getenv('BACKEND_PORT', 8001))

app = FastAPI(title="Prompt Evaluation Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
client = OpenAI(api_key=os.getenv("LITELLM_API_KEY"), base_url=f"{LITELLM_URL}/v1")


class ExperimentPayload(BaseModel):
    models: List[str]
    prompts: List[str]
    temperatures: List[float]
    max_tokens: List[int]
    timestamp: str


def calculate_cosine_similarity(text1, text2):
    emb1 = similarity_model.encode(text1, convert_to_tensor=False)
    emb2 = similarity_model.encode(text2, convert_to_tensor=False)
    return round(float(cosine_similarity([emb1], [emb2])[0][0]), 3)


def calculate_dot_product(text1, text2):
    emb1 = similarity_model.encode(text1, convert_to_tensor=False)
    emb2 = similarity_model.encode(text2, convert_to_tensor=False)
    return round(float(np.dot(emb1, emb2)), 3)


def generate_response(system_prompt, user_prompt, model, temperature=0.3, max_tokens=500):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=45.0
        )
        res = response.choices[0].message.content
        logger.info(f"Generated Message: {res[:50]}")
        return res, True
    except Exception as e:
        logger.error(f"[LLM ERROR] {model}: {e}")
        return f"Error: {str(e)}", False


def create_test_dataset():
    return [
        {"context": "PyTorch is an open-source machine learning library developed by Meta AI.",
         "query": "What is PyTorch and who developed it?", "category": "simple_factual"},
        {"context": "PyTorch was initially developed by Facebook's AI Research lab (FAIR) in 2016.",
         "query": "What company originally created PyTorch?", "category": "multi_hop"},
        {"context": "PyTorch provides automatic differentiation through its autograd package.",
         "query": "What is the latest version of PyTorch released in 2025?", "category": "out_of_context"},
        {"context": "PyTorch's autograd system records operations on tensors.",
         "query": "How does PyTorch implement automatic differentiation?", "category": "complex_technical"},
        {"context": "PyTorch tensors are similar to NumPy arrays but can run on GPUs.",
         "query": "What are PyTorch tensors and how do they differ from NumPy arrays?", "category": "multi_part"}
    ]


def evaluate_prompt_on_test_case(system_prompt, test_case, model, temperature, max_tokens, prompt_idx, model_idx):
    start_time = time.time()
    user_prompt_template = """**CONTEXT:**
{context}

**QUESTION:**
{query}

**INSTRUCTIONS:**
Answer the question using ONLY the information from the CONTEXT above.
If the context doesn't contain relevant information, respond with: "I don't have enough information..."

**ANSWER:**"""

    user_prompt = user_prompt_template.format(context=test_case['context'], query=test_case['query'])
    response_content, api_success = generate_response(system_prompt, user_prompt, model, temperature, max_tokens)

    response_time = time.time() - start_time
    query_response_similarity = calculate_cosine_similarity(test_case['query'], response_content)
    response_context_dotproduct = calculate_dot_product(response_content, test_case['context'])

    return {
        "model_index": model_idx,
        "prompt_index": prompt_idx,
        "model_name": model,
        "test_case": test_case['category'],
        "generated_response": response_content,
        "metrics": {
            "query_response_cosine_similarity": query_response_similarity,
            "response_context_dot_product": response_context_dotproduct,
            "response_time_seconds": round(response_time, 3),
            "response_length_words": len(response_content.split()),
            "api_success": api_success
        },
        "hyperparameters": {"temperature": temperature, "max_tokens": max_tokens}
    }


def normalize_metric(value, min_val, max_val, higher_is_better=True):
    if max_val == min_val: return 1.0
    normalized = (value - min_val) / (max_val - min_val)
    return normalized if higher_is_better else 1 - normalized


def run_experiment_sync(payload: dict, exp_id: str):
    start_time = time.time()
    logger.info(f"[ROCKET] {exp_id}: {len(payload['models'])} models x {len(payload['prompts'])} prompts")

    task = None
    try:
        # ClearML setup
        Task.set_credentials(
            api_host=os.getenv("CLEARML_API_HOST"),
            web_host=os.getenv("CLEARML_WEB_HOST"),
            files_host=os.getenv("CLEARML_FILES_HOST"),
            key=os.getenv("CLEARML_API_ACCESS_KEY"),
            secret=os.getenv("CLEARML_API_SECRET_KEY")
        )
        task = Task.init(project_name="RAG Prompt Evaluation", task_name=exp_id)
        logger.info(f"ClearML Task: {task.id}")

        models = payload["models"]
        prompts = payload["prompts"]
        temperatures = payload["temperatures"]
        max_tokens_list = payload["max_tokens"]
        test_dataset = create_test_dataset()

        # ✅ STEP 1: Run ALL combinations x 5 test cases
        all_results = []
        iteration = 0
        for model_idx, model in enumerate(models):
            for prompt_idx, system_prompt in enumerate(prompts):
                for temp in temperatures:
                    for max_tok in max_tokens_list:
                        combo_key = f"{model}_{prompt_idx}_{temp}_{max_tok}"
                        logger.info(f"Running combo: {combo_key}")

                        for test_case in test_dataset:
                            result = evaluate_prompt_on_test_case(
                                system_prompt, test_case, model, temp, max_tok, prompt_idx, model_idx
                            )
                            all_results.append(result)
                            iteration += 1

                            # Report to ClearML
                            task.logger.report_scalar(
                                "Cosine Similarity", f"{model}_P{prompt_idx + 1}",
                                result['metrics']['query_response_cosine_similarity'], iteration
                            )

        # ✅ STEP 2: Group by combination and calculate averages
        combo_averages = {}
        valid_results = [r for r in all_results if r['metrics']['api_success']]

        for result in valid_results:
            combo_key = f"{result['model_name']}_{result['prompt_index']}_{result['hyperparameters']['temperature']}_{result['hyperparameters']['max_tokens']}"

            if combo_key not in combo_averages:
                combo_averages[combo_key] = {
                    'cosine': [], 'dot': [], 'time': [], 'length': [], 'results': [],
                    'model_name': result['model_name'], 'prompt_index': result['prompt_index'],
                    'temperature': result['hyperparameters']['temperature'],
                    'max_tokens': result['hyperparameters']['max_tokens']
                }

            combo_averages[combo_key]['cosine'].append(result['metrics']['query_response_cosine_similarity'])
            combo_averages[combo_key]['dot'].append(result['metrics']['response_context_dot_product'])
            combo_averages[combo_key]['time'].append(result['metrics']['response_time_seconds'])
            combo_averages[combo_key]['length'].append(result['metrics']['response_length_words'])
            combo_averages[combo_key]['results'].append(result)

        # ✅ STEP 3: Calculate average metrics for each combo
        all_combo_avgs = {}
        for combo_key, combo in combo_averages.items():
            avg_metrics = {
                'query_response_cosine_similarity': np.mean(combo['cosine']),
                'response_context_dot_product': np.mean(combo['dot']),
                'response_time_seconds': np.mean(combo['time']),
                'response_length_words': np.mean(combo['length'])
            }
            all_combo_avgs[combo_key] = avg_metrics
            combo_averages[combo_key]['avg_metrics'] = avg_metrics

        # ✅ STEP 4: Calculate weighted scores for ALL combos
        metric_weights = {
            'query_response_cosine_similarity': 0.40,
            'response_context_dot_product': 0.30,
            'response_time_seconds': 0.20,
            'response_length_words': 0.10
        }

        combo_scores = {}
        for combo_key, avg_metrics in all_combo_avgs.items():
            score = 0
            total_weight = 0

            for metric_name, weight in metric_weights.items():
                if metric_name in avg_metrics:
                    all_metric_values = [all_combo_avgs[c][metric_name] for c in all_combo_avgs]
                    min_val, max_val = min(all_metric_values), max(all_metric_values)
                    higher_better = metric_name != 'response_time_seconds'
                    normalized = normalize_metric(avg_metrics[metric_name], min_val, max_val, higher_better)
                    score += normalized * weight
                    total_weight += weight

            combo_scores[combo_key] = round(score / total_weight if total_weight > 0 else 0, 3)

        # ✅ STEP 5: Get BEST combo and TOP 10
        best_combo_key = max(combo_scores, key=combo_scores.get)
        best_combo_data = combo_averages[best_combo_key]

        best_config = {
            "model_index": models.index(best_combo_data['model_name']) + 1,
            "model_name": best_combo_data['model_name'],
            "prompt_index": best_combo_data['prompt_index'] + 1,
            "prompt_name": f"Prompt {best_combo_data['prompt_index'] + 1}",
            "system_prompt_preview": prompts[best_combo_data['prompt_index']][:50] + "...",
            "full_system_prompt": prompts[best_combo_data['prompt_index']],
            "temperature": best_combo_data['temperature'],
            "max_tokens": best_combo_data['max_tokens'],
            "weighted_score": combo_scores[best_combo_key],
            "avg_cosine_similarity": round(best_combo_data['avg_metrics']['query_response_cosine_similarity'], 3),
            "avg_dot_product": round(best_combo_data['avg_metrics']['response_context_dot_product'], 3),
            "avg_response_time": round(best_combo_data['avg_metrics']['response_time_seconds'], 3),
            "avg_response_length": int(best_combo_data['avg_metrics']['response_length_words']),
            "total_tests": len(all_results),
            "valid_tests": len(valid_results),
            "total_combos": len(combo_averages)
        }

        # ✅ STEP 6: Create TOP 10 combinations (safe for <10)
        top_n_combos = min(10, len(combo_scores))
        all_combos_sorted = sorted(combo_scores.items(), key=lambda x: x[1], reverse=True)

        top_10_combinations = []
        for i, (combo_key, score) in enumerate(all_combos_sorted[:top_n_combos]):
            combo_data = combo_averages[combo_key]
            top_combo = {
                "rank": i + 1,
                "combo_key": combo_key,
                "test_case": f"{combo_data['model_name']} P{combo_data['prompt_index'] + 1}",
                "weighted_score": score,
                "model_name": combo_data['model_name'],
                "prompt_name": f"Prompt {combo_data['prompt_index'] + 1}",
                "temperature": combo_data['temperature'],
                "max_tokens": combo_data['max_tokens'],
                "metrics": {
                    "query_response_cosine_similarity": round(
                        combo_data['avg_metrics']['query_response_cosine_similarity'], 3),
                    "response_context_dot_product": round(combo_data['avg_metrics']['response_context_dot_product'], 3),
                    "response_time_seconds": round(combo_data['avg_metrics']['response_time_seconds'], 3),
                    "response_length_words": round(combo_data['avg_metrics']['response_length_words'], 3)
                }
            }
            top_10_combinations.append(top_combo)

        processing_time = time.time() - start_time
        results = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "best_config": best_config,
            "top_10_combinations": top_10_combinations,
            "total_combinations": len(combo_averages),
            "all_combo_scores": {k: float(v) for k, v in combo_scores.items()},
            "processing_time": round(processing_time, 2)
        }

        # Save and log
        task.upload_artifact("experiment_results", results)
        Path("experiment_results.json").write_text(json.dumps(results, indent=2))

        logger.info(
            f"[WINNER] {best_config['model_name']} + {best_config['prompt_name']} (T{best_config['temperature']}): {best_config['weighted_score']:.3f}")
        return results

    except Exception as e:
        logger.error(f"[ERROR] {exp_id} FAILED: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if task:
            task.close()


@app.post("/api/experiments")
async def start_experiment(payload: ExperimentPayload):
    exp_id = f"exp-{int(datetime.now().timestamp())}"
    total_combos = len(payload.models) * len(payload.prompts) * len(payload.temperatures) * len(payload.max_tokens)

    logger.info(f"[TARGET] {exp_id}: {total_combos} combos x 5 tests = {total_combos * 5} total tests")
    results = run_experiment_sync(payload.dict(), exp_id)

    return {
        "experiment_id": results["experiment_id"],
        "status": "completed",
        "total_combinations": results["total_combinations"],
        "peak_weighted_score": results["best_config"]["weighted_score"],
        "best_config": results["best_config"],
        "all_results": results["top_10_combinations"],
        "total_combos": results["total_combinations"]
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "prompt-evaluation-backend", "litellm_url": LITELLM_URL}

