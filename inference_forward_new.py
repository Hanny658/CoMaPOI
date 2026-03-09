"""
Forward Inference Module for CoMaPOI

This script performs forward inference for POI prediction using a multi-agent approach.
It coordinates three agents (Profiler, Forecaster, and Final_Predictor) to predict the next POI.
"""
import argparse
import json
import os
import time
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import agentscope
from evaluate import evaluate_poi_predictions
from utils import *
from agents import CustomDictDialogAgent, CustomReActAgent, CustomDialogAgent
from parser_tool import extract_predicted_pois
from tool.base_tools import *
from transformers import AutoTokenizer
from prompt_provider import PromptProvider
from agentscope.message import Msg
from agentscope.service import ServiceToolkit



def extract_and_clean_poi(prediction, top_k, max_item):
    """
    Extract and clean predicted POIs.

    Args:
        prediction: The prediction text to process
        top_k: Maximum number of POIs to return
        max_item: Maximum POI ID value

    Returns:
        list: Cleaned list of POI IDs
    """
    prediction = extract_predicted_pois(prediction, top_k)
    cleaned_prediction = clean_predicted_pois(prediction, max_item)
    return cleaned_prediction


def merge_valid_pois(existing_pois, new_pois, top_k):
    """
    Merge existing POIs with new POIs, ensuring no duplicates and respecting top_k limit.

    Args:
        existing_pois: List of existing POI IDs
        new_pois: List of new POI IDs to add
        top_k: Maximum number of POIs to return

    Returns:
        list: Merged list of POI IDs
    """
    # Convert all POIs to strings for consistency
    existing_pois = [str(poi) for poi in existing_pois]
    new_pois = [str(poi) for poi in new_pois]

    # Create a set of existing POIs for O(1) lookup
    existing_set = set(existing_pois)

    # Add new POIs that aren't already in the list
    for poi in new_pois:
        if poi not in existing_set and len(existing_pois) < top_k:
            existing_pois.append(poi)
            existing_set.add(poi)

    return existing_pois


def parse_reasoning_path(json_file_path, user_id):
    """
    Parse reasoning path from a JSON file and extract relevant components.

    Args:
        json_file_path: Path to the JSON file containing reasoning paths
        user_id: User ID to look up

    Returns:
        tuple: (long_term_profile, short_pattern_response, candidate_poi_list_agent1, candidate_poi_list_agent2)
    """
    try:
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Find data for the specified user_id
        user_data = next((item for item in data if str(item.get("user_id")) == str(user_id)), None)
        if not user_data or 'reasoning_path' not in user_data:
            raise ValueError(f"User ID {user_id} not found in JSON file or missing reasoning_path.")

        # Get reasoning_path
        reasoning_path = user_data['reasoning_path']

        # Extract content
        reasoning_data = {
            "long_term_profile": None,
            "short_pattern_response": None,
            "candidate_poi_list_agent1": None,
            "candidate_poi_list_agent2": None
        }

        # Extract each part from reasoning_path
        try:
            long_term_profile_start = reasoning_path.find("long_term_profile:")
            short_pattern_response_start = reasoning_path.find("short_pattern_response:")
            candidate_poi_list_agent1_start = reasoning_path.find("candidate_poi_list_agent1:")
            candidate_poi_list_agent2_start = reasoning_path.find("candidate_poi_list_agent2:")

            reasoning_data["long_term_profile"] = reasoning_path[long_term_profile_start:short_pattern_response_start].replace("long_term_profile: ", "").strip()
            reasoning_data["short_pattern_response"] = reasoning_path[short_pattern_response_start:candidate_poi_list_agent1_start].replace("short_pattern_response: ", "").strip()
            reasoning_data["candidate_poi_list_agent1"] = reasoning_path[candidate_poi_list_agent1_start:candidate_poi_list_agent2_start].replace("candidate_poi_list_agent1: ", "").strip()
            reasoning_data["candidate_poi_list_agent2"] = reasoning_path[candidate_poi_list_agent2_start:].replace("candidate_poi_list_agent2: ", "").strip()
        except Exception as parse_error:
            raise ValueError(f"Error parsing reasoning_path: {parse_error}")

        return reasoning_data["long_term_profile"], reasoning_data["short_pattern_response"], reasoning_data["candidate_poi_list_agent1"], reasoning_data["candidate_poi_list_agent2"]

    except Exception as e:
        print(f"Error while parsing reasoning_path: {e}")
        return None, None, None, None


def check_extra_information(long_term_profile, short_term_profile, candidate_poi_list_agent1, candidate_poi_list_agent2):
    """
    Check and extract information from JSON strings.

    Args:
        long_term_profile: Long-term profile JSON string
        short_term_profile: Short-term profile JSON string
        candidate_poi_list_agent1: Candidate POI list from agent 1 JSON string
        candidate_poi_list_agent2: Candidate POI list from agent 2 JSON string

    Returns:
        tuple: Extracted information from each input
    """
    def safe_json_load(data, key):
        try:
            return json.loads(data)[key]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    long_term_profile = safe_json_load(long_term_profile, "historical_profile")
    short_term_profile = safe_json_load(short_term_profile, "current_profile")
    candidate_poi_list_agent1 = safe_json_load(candidate_poi_list_agent1, "candidate_poi_list_from_profile")
    candidate_poi_list_agent2 = safe_json_load(candidate_poi_list_agent2, "refined_candidate_from_rag")

    return long_term_profile, short_term_profile, candidate_poi_list_agent1, candidate_poi_list_agent2


def get_candidate_poi_lists(args, candidate_poi_list_agent2, candidate_poi_list_agent1, rag_candidates):
    """
    Process candidate POI lists from both agents, removing duplicates and ensuring proper length.

    Args:
        args: Command line arguments
        candidate_poi_list_agent2: Candidate POI list from agent 2
        candidate_poi_list_agent1: Candidate POI list from agent 1
        rag_candidates: Additional candidates from RAG

    Returns:
        tuple: (processed_agent2_list, processed_agent1_list)
    """
    def process_poi_list(poi_list, seen_set):
        # Remove duplicates while preserving order
        unique_list = []
        for poi_id in poi_list:
            if poi_id not in seen_set:
                seen_set.add(poi_id)
                unique_list.append(poi_id)

        # Complete list with RAG candidates if needed
        rag_candidates_str = [str(int(rag_id)) for rag_id in rag_candidates if rag_id is not None]
        for rag_id in rag_candidates_str:
            if len(unique_list) >= args.num_candidate:  # Stop if we have enough
                break
            if rag_id not in seen_set:
                seen_set.add(rag_id)
                unique_list.append(rag_id)

        # Truncate to num_candidate length
        return unique_list[:args.num_candidate]

    # Process both candidate lists
    seen_agent2 = set()
    processed_agent2 = process_poi_list(candidate_poi_list_agent2, seen_agent2)

    seen_agent1 = set()
    processed_agent1 = process_poi_list(candidate_poi_list_agent1, seen_agent1)

    return processed_agent2, processed_agent1


def init_agents(args):
    """
    Initialize the language model agents for generation tasks.

    Args:
        args: Command line arguments

    Returns:
        tuple: (Profiler, Forecaster, Final_Predictor) - The initialized agents
    """
    base_url = f"http://{args.host}:{args.port}/v1"

    # Configure Agent 1 (Profiler)
    model_config_agent1 = {
        "config_name": f"{args.agent1_api}",
        "model_type": "openai_chat",
        "model_name": f"{args.agent1_api}",
        "api_key": "EMPTY",
        "client_args": {
            "base_url": base_url
        },
        "generate_args": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_tokens": args.agent1_max_tokens,
        }
    }

    # Configure Agent 2 (Forecaster)
    model_config_agent2 = {
        "config_name": f"{args.agent2_api}",
        "model_type": "openai_chat",
        "model_name": f"{args.agent2_api}",
        "api_key": "EMPTY",
        "client_args": {
            "base_url": base_url
        },
        "generate_args": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_tokens": args.agent2_max_tokens
        }
    }

    # Configure Agent 3 (Final_Predictor)
    model_config_agent3 = {
        "config_name": f"{args.agent3_api}",
        "model_type": "openai_chat",
        "model_name": f"{args.agent3_api}",
        "api_key": "EMPTY",
        "client_args": {
            "base_url": base_url
        },
        "generate_args": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n": args.n,
            "max_tokens": args.agent3_max_tokens
        }
    }

    # Initialize AgentScope with model configurations
    agentscope.init(model_configs=[model_config_agent1, model_config_agent2, model_config_agent3],
                    logger_level="CRITICAL",
                    use_monitor=False)

    # Create service toolkit and add tools
    service_toolkit = ServiceToolkit()
    service_toolkit.add(get_all_information_tool)

    # Initialize Profiler Agent
    Profiler = CustomDialogAgent(
        name="Profiler",
        sys_prompt="",
        model_config_name=args.agent1_api
    )

    # Initialize Forecaster Agent
    Forecaster = CustomDialogAgent(
        name="Forecaster",
        sys_prompt="",
        model_config_name=args.agent2_api
    )

    # Initialize Final_Predictor Agent
    Final_Predictor = CustomDialogAgent(
        name="Final_Predictor",
        sys_prompt="",
        model_config_name=args.agent3_api
    )

    return Profiler, Forecaster, Final_Predictor


def React_get_profile_information(args, agent, user_id):
    """
    Get profile information for a specific user.

    Args:
        args: Command line arguments
        agent: The agent to use for generation
        user_id: ID of the user to get profile for

    Returns:
        tuple: (summary, historical_information)
    """
    data = args.dataset
    summary, historical_information = get_profile_information(agent, user_id, data)
    return summary, historical_information


def process_single_profile(params):
    """
    Process a single user profile in a separate process.

    Args:
        params: Tuple containing (user_id, args)

    Returns:
        dict: User profile data or None if processing failed
    """
    i, args = params
    Profiler, _, _ = init_agents(args)
    Profiler.memory.clear()

    try:
        # Get user profile and historical information
        summary, historical_information = React_get_profile_information(args, Profiler, i)

        # Skip user if profile generation failed
        if not summary or not historical_information:
            return None

        # Prepare result
        result = {
            "user_id": i,
            "historical_information": historical_information,
        }

        # Verify result can be serialized
        try:
            import pickle
            pickle.dumps(result)
        except Exception as e:
            return None

        return result
    except Exception as e:
        return None


def React_process_and_save_profiles(args, output_file):
    """
    Process and save user profiles in parallel.

    Args:
        args: Command line arguments
        output_file: Output file path

    Returns:
        list: List of user profiles
    """
    start_point = args.start_point
    n = args.num_samples
    print(f"[INFO] Generating user profiles from {start_point} to {n-1}...")

    # Prepare parameters for parallel processing
    params_list = [(i, args) for i in range(start_point, n)]

    results = []
    with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
        futures = [executor.submit(process_single_profile, params) for params in params_list]

        # Use green progress bar with tqdm
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing profiles", colour="green"):
            result = future.result()
            if result:
                results.append(result)

    # Save results to file
    if results:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"[INFO] {len(results)} profiles have been saved to {output_file}.")
    else:
        print(f"[WARN] No profiles generated. Output file not created.")

    return results


def profiler_steps(Profiler, prompt_provider, user_id, current_trajectory, historical_trajectory):
    """
    Execute the Profiler agent steps to generate long-term profile and candidate POIs.

    Args:
        Profiler: The Profiler agent
        prompt_provider: Prompt provider for generating prompts
        user_id: User ID
        current_trajectory: Current trajectory data
        historical_trajectory: Historical trajectory data

    Returns:
        tuple: (long_term_profile, candidate_poi_list_agent1)
    """
    # Generate historical profile
    historical_profile_prompt = prompt_provider.get_a1p1_prompt(historical_trajectory)
    message_his_profile = Msg(name="Profiler", content=historical_profile_prompt, role="user")
    long_term_profile_msg = Profiler.reply(message_his_profile)
    long_term_profile = long_term_profile_msg.content

    # Generate candidate POI list
    merge_profiles_prompt = prompt_provider.get_a1p2_prompt(long_term_profile)
    message_merge_profiles = Msg(name="Profiler", content=merge_profiles_prompt, role="user")
    candidate_poi_list_msg = Profiler.reply(message_merge_profiles)
    candidate_poi_list_agent1 = candidate_poi_list_msg.content

    return long_term_profile, candidate_poi_list_agent1


def forecaster_steps(Forecaster, prompt_provider, user_to_candidate_map):
    """
    Execute the Forecaster agent steps to generate short-term pattern and refined candidates.

    Args:
        Forecaster: The Forecaster agent
        prompt_provider: Prompt provider for generating prompts
        user_to_candidate_map: Map of user IDs to candidate POIs

    Returns:
        tuple: (short_pattern_response, candidate_poi_list_agent2)
    """
    # Generate short-term pattern
    short_pattern_prompt = prompt_provider.get_a2p1_prompt()
    message_short_pattern = Msg(name="Forecaster", content=short_pattern_prompt, role="user")
    short_pattern_response = Forecaster.reply(message_short_pattern)
    short_pattern_response = short_pattern_response.content

    # Generate refined candidate list
    rag_candidates = user_to_candidate_map[prompt_provider.user_id]
    refine_candidates_prompt = prompt_provider.get_a2p2_prompt(short_pattern_response, rag_candidates)
    message_refine_candidates = Msg(name="Forecaster", content=refine_candidates_prompt, role="user")
    optimized_poi_list_msg = Forecaster.reply(message_refine_candidates)
    candidate_poi_list_agent2 = optimized_poi_list_msg.content

    return short_pattern_response, candidate_poi_list_agent2


def final_prediction_steps(Final_Predictor, prompt_provider, long_term_profile, short_term_profile, candidate_poi_list_agent1, candidate_poi_list_agent2):
    """
    Execute the Final_Predictor agent steps to generate the final prediction.

    Args:
        Final_Predictor: The Final_Predictor agent
        prompt_provider: Prompt provider for generating prompts
        long_term_profile: Long-term profile from Profiler
        short_term_profile: Short-term profile from Forecaster
        candidate_poi_list_agent1: Candidate POI list from Profiler
        candidate_poi_list_agent2: Candidate POI list from Forecaster

    Returns:
        tuple: (initial_prediction, final_prediction)
    """
    # Generate initial prediction
    prediction_prompt = prompt_provider.get_a3p1_prompt(long_term_profile, short_term_profile, candidate_poi_list_agent1, candidate_poi_list_agent2)
    message_init_prediction = Msg(name="Final_Predictor", content=prediction_prompt, role="user")
    initial_prediction_msg = Final_Predictor.reply(message_init_prediction)
    initial_prediction = initial_prediction_msg.content

    # Use initial prediction as final prediction (no reflection step)
    final_prediction = initial_prediction

    return initial_prediction, final_prediction


def validate_and_retry_sample(args, user_id, valid_poi_ids, Final_Predictor, prompt_provider):
    """
    Validate POI IDs and retry if invalid.

    Args:
        args: Command line arguments
        user_id: User ID
        valid_poi_ids: List of valid POI IDs
        Final_Predictor: The Final_Predictor agent
        prompt_provider: Prompt provider for generating prompts

    Returns:
        list: Validated POI IDs
    """
    max_item = args.max_item

    # Check if POI IDs are valid
    if len(valid_poi_ids) == args.top_k and \
       all(isinstance(poi_id, int) and 1 <= poi_id <= max_item for poi_id in valid_poi_ids) and \
       len(set(valid_poi_ids)) == len(valid_poi_ids):
        return valid_poi_ids  # Return original list if valid

    # Retry with Agent 3
    invalid_poi_ids = valid_poi_ids
    retry_prompt = prompt_provider.agent_retry_prompt(invalid_poi_ids)
    message_init_prediction = Msg(name="Agent3_Retry", content=retry_prompt, role="assistant")
    retry_prediction_msg = Final_Predictor.reply(message_init_prediction)
    retried_valid_poi_ids = extract_and_clean_poi(retry_prediction_msg.content, args.top_k, max_item)

    return retried_valid_poi_ids


def single_predict_save(params):
    """
    Predict POIs for a single sample using saved reasoning paths.

    Args:
        params: Tuple containing (selected_sample, args, user_to_candidate_map, historical_summary_list)

    Returns:
        tuple: (user_id, label, valid_poi_ids, init_valid_poi_ids, reasoning_path)
    """
    selected_sample, args, user_to_candidate_map, historical_summary_list = params
    user_id, label, current_trajectory = parse_user_and_trajectory(selected_sample.get('messages', []))

    # Set up paths
    args.saved_results_path = args.saved_results_path
    json_file_path = args.saved_results_path

    # Load reasoning path
    long_term_profile, short_pattern_response, candidate_poi_list_agent1, candidate_poi_list_agent2 = parse_reasoning_path(
        json_file_path, user_id)
    if not all([long_term_profile, short_pattern_response, candidate_poi_list_agent1, candidate_poi_list_agent2]):
        raise ValueError(f"Failed to load reasoning_path for user extracted from input: {user_id}.")

    # Apply ablation settings if specified
    if args.ab_type == 'profiler':
        long_term_profile = 'None'
        candidate_poi_list_agent1 = ['0'] * args.num_candidate
        candidate_poi_list_agent2 = ['0'] * args.num_candidate
    elif args.ab_type == 'forecaster':
        short_pattern_response = 'None'
        candidate_poi_list_agent2 = ['0'] * args.num_candidate
    elif args.ab_type == 'candidate':
        candidate_poi_list_agent1 = ['0'] * args.num_candidate
        candidate_poi_list_agent2 = ['0'] * args.num_candidate

    # Initialize agents
    Profiler, Forecaster, Final_Predictor = init_agents(args)
    Final_Predictor.memory.clear()

    # Initialize variables
    valid_poi_ids = []
    init_valid_poi_ids = []

    # Create prompt provider
    prompt_provider = PromptProvider(args, user_id, current_trajectory)

    # Get RAG candidates
    rag_candidates = user_to_candidate_map[user_id]

    # Generate predictions
    init_prediction, final_prediction = final_prediction_steps(Final_Predictor, prompt_provider, long_term_profile,
                                                             short_pattern_response, candidate_poi_list_agent1,
                                                             candidate_poi_list_agent2)

    # Process initial prediction
    init_predicted_pois = extract_and_clean_poi(init_prediction, top_k=args.top_k, max_item=args.max_item)
    init_valid_poi_ids = merge_valid_pois(valid_poi_ids, init_predicted_pois, args.top_k)

    # Process final prediction
    predicted_pois = extract_and_clean_poi(final_prediction, top_k=args.top_k, max_item=args.max_item)
    valid_poi_ids = merge_valid_pois(valid_poi_ids, predicted_pois, args.top_k)

    # Create reasoning path string
    reasoning_path = f"long_term_profile: {long_term_profile}, short_pattern_response: {short_pattern_response}, candidate_poi_list_agent1: [{candidate_poi_list_agent1}], candidate_poi_list_agent2: [{candidate_poi_list_agent2}], final_prediction: {final_prediction}"

    return user_id, label, valid_poi_ids[:args.top_k], init_valid_poi_ids[:args.top_k], reasoning_path


def single_predict(params):
    """
    Predict POIs for a single sample using the full agent pipeline.

    Args:
        params: Tuple containing (selected_sample, args, user_to_candidate_map, historical_summary_list)

    Returns:
        tuple: (user_id, label, valid_poi_ids, init_valid_poi_ids, reasoning_path)
    """
    selected_sample, args, user_to_candidate_map, historical_summary_list = params
    user_id, label, current_trajectory = parse_user_and_trajectory(selected_sample.get('messages', []))

    # Initialize agents
    Profiler, Forecaster, Final_Predictor = init_agents(args)

    # Clear agent memories
    Forecaster.memory.clear()
    Profiler.memory.clear()
    Final_Predictor.memory.clear()

    # Initialize variables
    valid_poi_ids = []
    init_valid_poi_ids = []

    # Create prompt provider
    prompt_provider = PromptProvider(args, user_id, current_trajectory)

    # Get historical summary
    his_summary = next((item for item in historical_summary_list if str(item["user_id"]) == user_id), None)

    # Apply ablation settings and run appropriate agent steps
    if args.ab_type == 'forecaster':
        # Run Profiler only
        long_term_profile, candidate_poi_list_agent1 = profiler_steps(Profiler, prompt_provider, user_id,
                                                                    current_trajectory, his_summary)
        candidate_poi_list_agent1 = extract_and_clean_poi(candidate_poi_list_agent1, top_k=args.num_candidate,
                                                        max_item=args.max_item)

        # Skip Forecaster
        short_pattern_response = 'None'
        candidate_poi_list_agent2 = ['0'] * args.num_candidate

    elif args.ab_type == 'profiler':
        # Run Profiler but ignore results
        long_term_profile, candidate_poi_list_agent1 = profiler_steps(Profiler, prompt_provider, user_id,
                                                                    current_trajectory, his_summary)
        candidate_poi_list_agent1 = extract_and_clean_poi(candidate_poi_list_agent1, top_k=args.num_candidate,
                                                        max_item=args.max_item)
        long_term_profile = 'None'
        candidate_poi_list_agent1 = ['0'] * args.num_candidate

        # Run Forecaster
        short_pattern_response, candidate_poi_list_agent2 = forecaster_steps(Forecaster, prompt_provider,
                                                                           user_to_candidate_map)
        candidate_poi_list_agent2 = extract_and_clean_poi(candidate_poi_list_agent2, top_k=args.num_candidate,
                                                         max_item=args.max_item)

    elif args.ab_type == 'candidate':
        # Run Profiler
        long_term_profile, candidate_poi_list_agent1 = profiler_steps(Profiler, prompt_provider, user_id,
                                                                    current_trajectory, his_summary)
        candidate_poi_list_agent1 = ['0'] * args.num_candidate

        # Run Forecaster
        short_pattern_response, candidate_poi_list_agent2 = forecaster_steps(Forecaster, prompt_provider,
                                                                           user_to_candidate_map)
        candidate_poi_list_agent2 = ['0'] * args.num_candidate

    else:
        # Run full pipeline
        long_term_profile, candidate_poi_list_agent1 = profiler_steps(Profiler, prompt_provider, user_id,
                                                                    current_trajectory, his_summary)
        candidate_poi_list_agent1 = extract_and_clean_poi(candidate_poi_list_agent1, top_k=args.num_candidate,
                                                        max_item=args.max_item)

        short_pattern_response, candidate_poi_list_agent2 = forecaster_steps(Forecaster, prompt_provider,
                                                                           user_to_candidate_map)
        candidate_poi_list_agent2 = extract_and_clean_poi(candidate_poi_list_agent2, top_k=args.num_candidate,
                                                         max_item=args.max_item)

    # Get RAG candidates
    rag_candidates = user_to_candidate_map[user_id]

    # Generate predictions
    init_prediction, final_prediction = final_prediction_steps(Final_Predictor, prompt_provider, long_term_profile,
                                                             short_pattern_response, candidate_poi_list_agent1,
                                                             candidate_poi_list_agent2)

    # Process initial prediction
    init_predicted_pois = extract_and_clean_poi(init_prediction, top_k=args.top_k, max_item=args.max_item)
    init_valid_poi_ids = merge_valid_pois(valid_poi_ids, init_predicted_pois, args.top_k)

    # Process final prediction
    predicted_pois = extract_and_clean_poi(final_prediction, top_k=args.top_k, max_item=args.max_item)
    valid_poi_ids = merge_valid_pois(valid_poi_ids, predicted_pois, args.top_k)

    # Create reasoning path string
    reasoning_path = f"long_term_profile: {long_term_profile}, short_pattern_response: {short_pattern_response}, candidate_poi_list_agent1: [{candidate_poi_list_agent1}], candidate_poi_list_agent2: [{candidate_poi_list_agent2}], final_prediction: {final_prediction}"

    return user_id, label, valid_poi_ids[:args.top_k], init_valid_poi_ids[:args.top_k], reasoning_path


class ForwardInferenceProcessor:
    """
    Main class for forward inference processing.
    This class handles the initialization of agents and processing of data for forward inference.
    """

    def __init__(self, args):
        """
        Initialize the ForwardInferenceProcessor with command line arguments.

        Args:
            args: Command line arguments parsed by argparse
        """
        self.args = args

    def parallel_predict(self):
        """
        Run parallel prediction using multiple processes.
        Sets up the environment, processes samples in parallel, and saves results.

        Returns:
            str: Path to the generated results file
        """
        args = self.args
        n, dataset, top_k = args.num_samples, args.dataset, args.top_k

        # Set up file paths
        results_path = f'results/{dataset}/{args.save_name}'
        data_path = f'dataset_all/{dataset}/{args.mode}'

        os.makedirs(results_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)

        output_json = f'{results_path}/poi_predictions.json'
        metrics_txt = f'{results_path}/metrics.txt'
        metrics_csv = f'{results_path}/metrics.csv'

        candidate_output_json = data_path + f"/{args.dataset}_{args.mode}_candidates.jsonl"

        # Load samples
        samples = []
        with open(f'dataset_all/{dataset}/{args.mode}/{dataset}_{args.mode}.jsonl', 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        num_samples = min(n, len(samples))
        samples = samples[:num_samples]

        print(f"Processing {num_samples} samples with {args.batch_size} parallel workers...")

        # Load or generate historical summaries
        historical_distribution_path = f'dataset_all/{args.dataset}/{args.dataset}_historical_summary.jsonl'
        if os.path.exists(historical_distribution_path):
            print(f"[INFO] Loading historical profiles from {historical_distribution_path}")
            with open(historical_distribution_path, 'r', encoding='utf-8') as f:
                historical_summary_list = [json.loads(line) for line in f]
        else:
            print(f"[INFO] Generating historical profiles...")
            historical_summary_list = React_process_and_save_profiles(args, historical_distribution_path)

        # Load candidate list
        user_to_candidate_map = load_candidate_list(candidate_output_json)

        # Prepare parameters for parallel processing
        params_list = []
        for i in range(args.start_point, n):
            selected_sample = samples[i % num_samples]
            params = (selected_sample, args, user_to_candidate_map, historical_summary_list)
            params_list.append(params)

        all_predictions = {}

        # Run parallel processing
        with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
            if args.load_pf_output:
                futures = [executor.submit(single_predict_save, params) for params in params_list]
            else:
                futures = [executor.submit(single_predict, params) for params in params_list]

            # Use green progress bar with tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Predicting POIs", colour="green"):
                user_id, label, valid_poi_ids, init_valid_poi_ids, reasoning_path = future.result()

                all_predictions[user_id] = {
                    "user_id": user_id,
                    "label": label,
                    "reasoning_path": reasoning_path,
                    "predicted_poi_ids": valid_poi_ids,
                    "init_valid_poi_ids": init_valid_poi_ids,
                }

                # Save and evaluate at intervals
                if len(all_predictions) % args.test_interval == 0:
                    print(f"\n[INFO] Completed {len(all_predictions)} samples. Saving interim results and evaluating.")
                    interim_output_json = f'{results_path}/interim_poi_predictions_{len(all_predictions)}.json'

                    # Save interim predictions
                    with open(interim_output_json, 'w', encoding='utf-8') as f:
                        json.dump(list(all_predictions.values()), f, ensure_ascii=False, indent=4)

                    # Evaluate interim results
                    evaluate_poi_predictions(args, interim_output_json, top_k, metrics_txt, metrics_csv, key='predicted_poi_ids')

        # Save final results
        print("\n[INFO] Processing complete. Saving final prediction results.")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(list(all_predictions.values()), f, ensure_ascii=False, indent=4)
        print(f"[INFO] All final prediction results saved to: {output_json}")

        # Final evaluation
        evaluate_poi_predictions(args, output_json, top_k, metrics_txt, metrics_csv, key='predicted_poi_ids')
        print("[INFO] Final evaluation metrics saved.")

        return output_json


def main():
    """
    Main function to parse arguments and run the forward inference process.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="POI Prediction Agent - Forward Inference")
    parser.add_argument('--num_samples', type=int, default=0, help='Number of samples to process')
    parser.add_argument('--dataset', type=str, default='nyc', choices=['nyc', 'tky', 'ca'], help='Dataset to use')
    parser.add_argument('--top_k', type=int, default=10, help='Top K predictions')
    parser.add_argument('--model', type=str, default='qwen2.5-1.5b-instruct', help='Base model to use')
    parser.add_argument('--api_type', type=str, default="gpt", help='API type for model')
    parser.add_argument('--max_item', type=int, default=5091, help='Maximum POI ID value')
    parser.add_argument('--max_retry', type=int, default=1, help='Maximum number of retries')
    parser.add_argument('--start_point', type=int, default=0, help='Starting point for processing')
    parser.add_argument('--test_interval', type=int, default=200, help='Test interval for saving and evaluating')
    parser.add_argument('--batch_size', type=int, default=8, help='Number of concurrent processes')
    parser.add_argument('--mode', type=str, default='test', help='Mode (train/test)')
    parser.add_argument('--save_name', type=str, default='N1', help='Save ID (N1-N...; T1-T...; C1-C...)')
    parser.add_argument('--agent1_api', type=str, default='agent1', help='API name for Agent 1 (Profiler)')
    parser.add_argument('--agent2_api', type=str, default='agent2', help='API name for Agent 2 (Forecaster)')
    parser.add_argument('--agent3_api', type=str, default='agent3', help='API name for Agent 3 (Final_Predictor)')
    parser.add_argument('--num_candidate', type=int, default=25, help='Number of candidate POIs')
    parser.add_argument('--store_save_name', action="store_true", help='Manually provide storage name')
    parser.add_argument('--host', type=str, default='localhost', help='Host for API server')
    parser.add_argument('--port', type=int, default=7863, help='Port for API server')
    parser.add_argument('--agent1_max_tokens', type=int, default=256, help='Max tokens for Agent 1')
    parser.add_argument('--agent2_max_tokens', type=int, default=256, help='Max tokens for Agent 2')
    parser.add_argument('--agent3_max_tokens', type=int, default=256, help='Max tokens for Agent 3')
    parser.add_argument('--sub_file', type=str, default='ablation', help='Sub-directory for results')
    parser.add_argument('--load_pf_output', action="store_true", help='Load pre-generated profiles')
    parser.add_argument('--saved_results_path', type=str, default='none', help='Path to saved results')
    parser.add_argument('--op_str', type=str, default='none', help='Operation string')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=1, help='Top-p for generation')
    parser.add_argument('--n', type=int, default=1, help='Number of generations')
    parser.add_argument('--prompt_format', type=str, default="json", help='Prompt format')
    parser.add_argument('--ab_type', type=str, default="none", help='Ablation type')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()
    dataset = args.dataset

    # Set dataset-specific parameters
    args.max_item = {"nyc": 5091, "tky": 7851, "ca": 13630}.get(dataset, 5091)
    if args.num_samples == 0:
        args.num_samples = {"nyc": 988, "tky": 2206, "ca": 1818}.get(dataset, 988)

    # Set save name if not manually provided
    if not args.store_save_name:
        args.save_name = f"{args.op_str}/[Forward_Inference_{args.dataset}_{args.model}_{args.batch_size}_{args.num_candidate}_agent1_api_{args.agent1_api}_agent2_api_{args.agent2_api}_agent3_api_{args.agent3_api}]"

    print("Starting forward inference with arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Create processor and run parallel prediction
    processor = ForwardInferenceProcessor(args)
    processor.parallel_predict()


if __name__ == "__main__":
    if os.name == 'nt':
        # Windows platform needs protection at the main entry point
        import multiprocessing
        multiprocessing.freeze_support()
    main()
