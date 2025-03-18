from openai import OpenAI
import json
from datasets import load_from_disk, Dataset
from document_store import DocumentStore, E5EmbeddingFunction
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import torch
from tqdm import tqdm
import os
from functools import partial
import argparse
from openai.types.chat.chat_completion_message import ChatCompletionMessage
import pickle

def get_mrr(condition, rollout_result):
    """Calculate Mean Reciprocal Rank for a single query.
    
    Args:
        condition (dict): Contains the ground truth information including correct_id
        rollout_result (list): List of messages containing retrieved_ids
    """
    retrieved_ids = list(
        map(
            lambda x: json.loads(x.get('content', None)).get('ids', None), 
            filter(
                lambda x: x.get('role', None) == 'ipython' or x.get('role', None) == 'tool', 
                rollout_result
            )
        )
    )
    
    if len(retrieved_ids) == 0:
        return 0.0
    
    mrr = 0.0
    correct_id = condition.get('document_id')
    for retrieval in retrieved_ids:
        _retrieval = [r.split("_chunk_")[0] for r in retrieval]
        
        if correct_id in _retrieval:
            rank = _retrieval.index(correct_id) + 1
            if 1.0 / rank > mrr:
                mrr = 1.0 / rank
    return mrr

def get_answer_similarity(embedding_function, condition, rollout_result):
    """Calculate semantic similarity between model answer and ground truth.
    
    Args:
        condition (dict): Contains the ground truth and embedding function
        rollout_result (list): List of messages containing model's answer
    """

    # Make sure last message generated and not tool call
    if rollout_result[-1].get('role', None) != 'assistant':
        return 0.0
    if rollout_result[-1].get('tool_calls', None) is not None:
        return 0.0
    
    model_answer = rollout_result[-1].get('content', None).split('</think>')
    if len(model_answer) > 0:
        model_answer = model_answer[-1]
    else:
        return 0.0

    # medquad
    ground_truth = condition.get('answer', None)
    # tldr
    # ground_truth = condition.get('completion', None)
    
    if model_answer is None:
        return 0.0
    # Get embeddings
    with torch.no_grad():
        # prefix with 'query: ' according to https://huggingface.co/intfloat/e5-large
        embeddings = embedding_function([
            'query: ' + model_answer,
            'query: ' + ground_truth
        ])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]
    
    return float(max(0, similarity))

def get_completion_length(condition, rollout_result):
    """
    Calculate the length of the completion.
    
    Args:
        condition (dict): Contains the ground truth and embedding function
        rollout_result (list): List of messages containing model's answer
    """
    # Make sure last message generated and not tool call
    if rollout_result[-1].get('role', None) != 'assistant':
        return 0.0
    if rollout_result[-1].get('tool_calls', None) is not None:
        return 0.0
    
    model_answer = rollout_result[-1].get('content', None)
    if model_answer is None:
        return 0.0

    if not '</think>' in model_answer:
        return -100
    
    reward = -len(model_answer.split('</think>')[1].split(' '))/20 # target length of 20 words
    return reward

def get_format_reward(condition, rollout_result):
    """Calculate reward based on format compliance.
    
    Args:
        condition (dict): Contains format requirements and constraints
        rollout_result (list): List of messages to check for format compliance
    """
    generated_responses = list(filter(
        lambda x: x.get('role', None) == 'assistant',
        rollout_result
    ))

    # no tool calls or ipython calls are bad
    tool_calls = list(filter(
        lambda x: x.get('role', None) == 'ipython' or x.get('role', None) == 'tool', 
        rollout_result
    ))

    if len(tool_calls) == 0:
        return 0.0

    num_correct_format = 0
    
    for response in generated_responses:
        content = response.get('content', None)
        if content is None:
            continue
        
        # Check if content matches format requirements
        if '</think>' in content:
            num_correct_format += 1

    return num_correct_format / len(generated_responses)

def get_reward_functions(embedding_function):
    """Get the reward functions with initialized embedding function.

    Returns:
        Dictionary of reward functions
    """
    return {
        "mrr": get_mrr,
        "answer_similarity": partial(get_answer_similarity, embedding_function),
        "format": get_format_reward
    }


def get_tldr_reward(embedding_function):
    return {
        "answer_similarity": partial(get_answer_similarity, embedding_function),
        "length": get_completion_length
    }


def calculate_rewards(condition, rollout_results, reward_functions, weight_scheme="uniform"):
    """Calculate rewards for a group of rollout results.
    
    Args:
        rollout_results: List of dictionaries containing metric scores
        reward_functions: Dictionary of reward functions in the form {name: function(metrics_dict)}
                         where metrics_dict contains the metrics from rollout_results
        weight_scheme: One of "uniform", "std", "pca", or dict of {name: weight}
    
    Returns:
        List of reward values for each rollout
    
    Raises:
        ValueError: If reward_functions is empty or None
    """
    if not reward_functions:
        raise ValueError("reward_functions must be provided and cannot be empty")
    
    # Calculate rewards for each function
    rewards_by_function = {}
    for name, func in reward_functions.items():
        rewards_by_function[name] = [
            func(condition=condition, rollout_result=result)
            for result in rollout_results
        ]
    
    # Calculate weights based on scheme
    if isinstance(weight_scheme, dict):
        weights = weight_scheme
    elif weight_scheme == "std":
        total_std = sum(np.std(rewards) for rewards in rewards_by_function.values())
        weights = {
            name: np.std(rewards) / total_std
            for name, rewards in rewards_by_function.items()
        }
    elif weight_scheme == "pca":
        # Convert rewards_by_function to a matrix where rows are samples and columns are features (reward functions)
        reward_names = list(rewards_by_function.keys())
        reward_matrix = np.column_stack([rewards_by_function[name] for name in reward_names])
        
        # Standardize the data (mean=0, std=1) for each feature
        reward_matrix_std = (reward_matrix - np.mean(reward_matrix, axis=0)) / (np.std(reward_matrix, axis=0) + 1e-8)
        
        # Apply PCA to project to 1 dimension (maximum variance direction)
        pca = PCA(n_components=1)
        projected_rewards = pca.fit_transform(reward_matrix_std).flatten()
        
        # Use the projected values directly as rewards
        # No need for weights since we're returning the PCA result directly
        return {
            "advantages": ((projected_rewards - np.mean(projected_rewards)) / (np.std(projected_rewards) + 1e-8)).tolist(),
            "rewards": projected_rewards.tolist(),
            "weights": {name: float(component) for name, component in zip(reward_names, pca.components_[0])},
            "rewards_by_function": rewards_by_function,
            "std": float(np.std(projected_rewards))
        }
    elif weight_scheme == "l2norm":
        rewards = [np.array(rewards) for rewards in rewards_by_function.values()]
        rewards = [np.square(r) for r in rewards]
        rewards = np.sqrt(np.sum(rewards, axis=0))
        return {
            "advantages": ((rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)).tolist(),
            "rewards": rewards.tolist(),
            "weights": {name: 1.0 for name in reward_functions.keys()},
            "rewards_by_function": rewards_by_function,
            "std": float(np.std(rewards))
        }
    else:  # uniform
        weights = {name: 1.0/len(reward_functions) for name in reward_functions.keys()}
    
    # Calculate final weighted rewards
    final_rewards = np.zeros(len(rollout_results))
    for name, rewards in rewards_by_function.items():
        final_rewards += np.array(rewards) * weights[name]
    
    return {
        "advantages": ((final_rewards - np.mean(final_rewards)) / (np.std(final_rewards) + 1e-8)).tolist(),
        "rewards": final_rewards.tolist(),
        "weights": weights,
        "rewards_by_function": rewards_by_function,
        "std": float(np.std(final_rewards))
    }

def parse_rollout(grouped_rollout, rewards_info, filter_by_mean=False):
    """Create a dataset from rollout results and rewards.
    
    Args:
        grouped_results: List of dictionaries containing example data and rollouts
        rewards_info: List of reward information from calculate_rewards
        filter_by_mean: Boolean indicating whether to filter rollouts by mean reward
        
    Returns:
        Dictionary containing messages, rewards, and metrics
    """
    all_messages = []
    all_advantages = []
    all_metrics = []
    
    for group, reward_info in zip(grouped_rollout, rewards_info):
        rollouts = group["rollouts"]
        original_reward_by_function = reward_info["rewards_by_function"]
        
        # Store individual metric values for each rollout BEFORE filtering
        for i in range(len(rollouts)):
            metrics = {
                name: values[i]
                for name, values in original_reward_by_function.items()
            }
            metrics["std"] = reward_info["std"]
            all_metrics.append(metrics)

        if filter_by_mean:
            
            # Filter out rollouts with rewards not all above or below mean
            reward_by_function = reward_info["rewards_by_function"]
            reward_mean = {name: np.mean(values) for name, values in reward_by_function.items()}
            
            # Create a filtered list of rollouts, advantages, and their indices
            filtered_rollouts = []
            filtered_advantages = []
            filtered_indices = []
            
            for i, rollout in enumerate(rollouts):
                # Check if all rewards for this rollout are consistently above or below their respective means
                all_above_mean = all(reward_by_function[name][i] >= reward_mean[name] for name in reward_by_function.keys())
                all_below_mean = all(reward_by_function[name][i] <= reward_mean[name] for name in reward_by_function.keys())
                # Keep rollouts that are consistently above or below mean across all reward functions
                if all_above_mean or all_below_mean:
                    filtered_rollouts.append(rollout)
                    filtered_advantages.append(reward_info["advantages"][i])
                    filtered_indices.append(i)
            
            # Replace original lists with filtered versions
            rollouts = filtered_rollouts
            reward_info["advantages"] = filtered_advantages
            
            # Filter the rewards_by_function to only include the kept indices
            reward_by_function = {name: [reward_by_function[name][i] for i in filtered_indices] for name in reward_by_function.keys()}
            reward_info["rewards_by_function"] = reward_by_function
            
        # Store rollout messages
        all_messages.extend(rollouts)
        all_advantages.extend(reward_info["advantages"])
    
    return {
        'messages': all_messages,
        'advantages': all_advantages,
        'metrics': all_metrics
    }

def message_to_dict(message):
    if isinstance(message, dict):
        return message
    else:
        if message.tool_calls:
            return {
                'role':'assistant',
                'content':message.content,
                'tool_calls':[
                    {
                        'function': {
                            'arguments': json.loads(tool_call.function.arguments),
                            'name': tool_call.function.name
                        }
                    } for tool_call in message.tool_calls
                ]
            }
        else:
            return {
                'role':'assistant',
                'content':message.content
            }

def run_llm_rollout(
    dataset,
    doc_store,
    num_rollouts=64,
    model_path="grpo",
    vllm_port=8000
):
    """Run LLM rollout on the dataset.
    
    Args:
        dataset: HuggingFace dataset containing examples
        doc_store: Document store instance for search
        num_rollouts: Maximum number of rollouts per example
        model_path: Path to the model to use for rollouts
        vllm_port: Port number for vLLM server
    Returns:
        List of dictionaries containing example data and rollout results
    """

    client = OpenAI(
        base_url=f'http://localhost:{vllm_port}/v1',
        api_key="EMPTY",
    )
    
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search documents in the document store",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query to search documents"},
                },
                "required": ["query"]
            }
        }
    }]
    # medquad
    system_prompt = """### Role
You are an assistant that only answers questions about the medical document store.

### Instructions
You must use the search tool once and only once to find relevant documents to the question. 
You must perform step by step planning and reasoning every time before using tool and giving response to user.

### Planning and Reasoning Format
Every response must begin with <think> tag, followed with step by step planning and reasoning, and then a </think> tag.
After planning and reasoning, start your answer or tool calls."""
    # tldr
#     system_prompt = """### Role
# # You are an assistant that create short and concise summary of the inputted passage.

# ### Instructions
# # Every response must begin with <think> tag, followed with step by step planning and reasoning, and then a </think> tag."""
    
    # Process examples
    grouped_rollout = []
    
    num_examples = len(dataset)
    for i in tqdm(range(0, num_examples), total=num_examples, desc="Running rollouts"):
        example = dataset[i]
        rollout = []
        
        # Run rollouts for this example
        while len(rollout) < num_rollouts:
            messages = [
                {
                    'role':'system',
                    'content':system_prompt
                },
                {
                    'role':"user",
                    'content':example['question'], # medquad
                    # 'content': example['prompt'], # tldr
                }
            ]
            
            error_count = 0
            for _ in range(3):
                try:
                    if isinstance(messages[-1], ChatCompletionMessage) and messages[-1].tool_calls:
                        messages.append({
                            'role':'tool',
                            'content':json.dumps(
                                doc_store.search(
                                query=json.loads(messages[-1].tool_calls[0].function.arguments).get('query', ''),
                                n_results=5
                            )
                        )
                    })
                    if isinstance(messages[-1], dict):
                        chat_completion = client.chat.completions.create(
                            model=model_path,
                            messages=messages,
                            tools=tools, # tldr has no tools
                        )
                        response = chat_completion.choices[0].message
                        messages.append(response)
                except Exception as e:
                    error_count += 1
                    break

            # Store rollout result
            if error_count == 0:
                rollout.append([message_to_dict(x) for x in messages])

        
        # Store results for this example
        grouped_rollout.append({
            "example": example,
            "rollouts": rollout
        })
    
    return grouped_rollout

def main():
    parser = argparse.ArgumentParser(description='Run LLM rollouts')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--num_rollouts', type=int, default=64, help='Number of rollouts per example')
    parser.add_argument('--model', type=str, default='/workspace/Qwen2.5-7B-Instruct-qlora', help='Path to the model')
    parser.add_argument('--vllm_port', type=int, default=8000, help='Port to use for vLLM server')
    args = parser.parse_args()

    # Load dataset
    dataset = load_from_disk(args.dataset)
    
    # Initialize document store
    doc_store = DocumentStore(
        collection_name="documents",
        chroma_db_path="./chroma_db",
        bm25_index_path="./bm25_index",
        document_content_dir="./document_content",
        device=args.device
    )
    
    # 1. Run rollouts
    grouped_rollout = run_llm_rollout(
        dataset=dataset,
        doc_store=doc_store,
        num_rollouts=args.num_rollouts,
        model_path=args.model,
        vllm_port=args.vllm_port
    )
    
    # 2. Calculate rewards
    reward_functions = get_reward_functions(embedding_function=doc_store.embedding_function)
    # reward_functions = get_tldr_reward(embedding_function=doc_store.embedding_function)
    rewards_info = [
        calculate_rewards(
            condition=group["example"],
            rollout_results=group["rollouts"],
            reward_functions=reward_functions,
            weight_scheme={"mrr": 0.0, "answer_similarity": 1.0, "format": 1.0}
            # weight_scheme={"answer_similarity": 4.0, "length": 1.0}, # tldr
        )
        for group in grouped_rollout
    ]

    # 3. Prase results
    rollout_result = parse_rollout(grouped_rollout, rewards_info)

    # 4. Save results using pickle
    output_path = os.path.join(args.dataset, "rollout_results.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(rollout_result, f)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open(os.path.join('std.err'), 'w') as f:
            f.write(str(e))