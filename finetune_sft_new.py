"""
CoMaPOI Model Fine-tuning Module

This script performs supervised fine-tuning (SFT) for language models used in the CoMaPOI system.
It supports fine-tuning for individual agents (Profiler, Forecaster, Final_Predictor) or a merged model.
"""
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import argparse
import os
import json
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
# Add NumPy 2.0 compatibility patch
import os
import numpy as np

# Explicitly disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Add compatibility for NumPy 2.0
if not hasattr(np, 'float_'):
    np.float_ = np.float64

from trl import SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

from config import DATASET_MAX_ITEM, FINETUNE_DATA_ROOT, FINETUNE_RESULTS_ROOT


# Helper functions for data processing
def prepare_sample_text(example: Dict[str, Any]) -> str:
    """
    Format a dataset sample into text suitable for training.

    Args:
        example: Dataset example with 'messages' field

    Returns:
        str: Formatted text with question and answer
    """
    # Extract assistant content
    assistant_content = next(
        (msg['content'] for msg in example['messages'] if msg['role'] == 'assistant'),
        None
    )

    if assistant_content:
        # Remove prefixes like "Answer: "recent_mobility_analysis": " or "Answer: "historical_profile": "
        assistant_content = re.sub(
            r'^Answer:\s*"?(recent_mobility_analysis|historical_profile)"?:\s*"?',
            '',
            assistant_content,
            flags=re.DOTALL
        ).strip(' "')

    # Extract POI ID
    try:
        assistant_data = json.loads(assistant_content)
        if isinstance(assistant_data, int):
            poi_id = assistant_data
        elif isinstance(assistant_data, dict):
            poi_id = assistant_data.get("next_poi_id", "unknown")
        else:
            poi_id = "unknown"
    except json.JSONDecodeError:
        poi_id = "unknown"

    # Extract user content
    user_content = next(
        (msg['content'] for msg in example['messages'] if msg['role'] == 'user'),
        ""
    )
    user_content = re.sub(r"OUTPUT FORMAT:.*", "", user_content, flags=re.DOTALL).strip()

    # Format the text
    text = f"Question: {user_content}\n\nAnswer: {assistant_content}"
    return text


def count_tokens(example: Dict[str, Any], tokenizer) -> int:
    """
    Count the number of tokens in a sample.

    Args:
        example: Dataset example
        tokenizer: Tokenizer to use for counting

    Returns:
        int: Number of tokens
    """
    text = prepare_sample_text(example)
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=False)
    token_count = tokenized_text["input_ids"].shape[1]
    return token_count


def chars_token_ratio(dataset, tokenizer, args) -> Tuple[float, int]:
    """
    Estimate the average number of characters per token in the dataset.

    Args:
        dataset: Dataset to analyze
        tokenizer: Tokenizer to use
        args: Command line arguments

    Returns:
        Tuple[float, int]: (characters per token ratio, maximum token count)
    """
    total_characters, total_tokens = 0, 0
    max_token_count = 0
    nb_examples = len(dataset)

    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        token_count = count_tokens(example, tokenizer)
        text = prepare_sample_text(example)
        max_token_count = max(max_token_count, token_count)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    print(f"Maximum token count in dataset: {max_token_count}")
    return total_characters / total_tokens, max_token_count


def print_trainable_parameters(model) -> None:
    """
    Print the number of trainable parameters in the model.

    Args:
        model: Model to analyze
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable parameters: {trainable_params} || Total parameters: {all_param} || Trainable percentage: {100 * trainable_params / all_param:.2f}%"
    )


class DataProcessor:
    """
    Class for processing and preparing training data.
    """

    @staticmethod
    def process_and_split_jsonl(input_file_path: str, cleaned_train_file_path: str,
                               cleaned_test_file_path: str, test_size: int = 100) -> None:
        """
        Process a JSONL file, ensure all 'content' fields are strings, check if each sample
        contains 'user' and 'assistant' content, and split the data into training and test sets.

        Args:
            input_file_path: Path to the original JSONL file
            cleaned_train_file_path: Path to save the cleaned training JSONL file
            cleaned_test_file_path: Path to save the cleaned test JSONL file
            test_size: Number of samples for the test set
        """
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"Input file does not exist: {input_file_path}")

        cleaned_samples = []

        with open(input_file_path, 'r', encoding='utf-8') as fin:
            line_num = 0
            for line in fin:
                line_num += 1
                try:
                    data = json.loads(line)

                    # Check if 'messages' key exists and is a list
                    if 'messages' not in data or not isinstance(data['messages'], list):
                        continue

                    messages = data['messages']

                    # Check if it contains at least one 'user' and one 'assistant' message
                    has_user = False
                    has_assistant = False
                    assistant_label = None
                    for message in messages:
                        if 'role' in message and 'content' in message:
                            # Ensure 'content' is a string
                            message['content'] = str(message['content'])
                            if message['role'] == 'user':
                                has_user = True
                            elif message['role'] == 'assistant':
                                has_assistant = True
                                assistant_label = message['content']  # Get 'assistant' content as label

                    if not (has_user and has_assistant):
                        continue

                    # Check if 'assistant_label' is empty
                    if assistant_label is None or assistant_label == "":
                        continue

                    # If the sample passes all checks, add it to the cleaned samples list
                    cleaned_samples.append(data)

                except json.JSONDecodeError:
                    continue

        total_cleaned = len(cleaned_samples)
        if total_cleaned == 0:
            raise ValueError("No valid samples were cleaned. Please check the input file.")

        print(f"Total of {total_cleaned} valid samples cleaned.")

        # Ensure test set size doesn't exceed total sample count
        test_size = min(test_size, total_cleaned)

        # Shuffle samples randomly
        random.shuffle(cleaned_samples)

        # Split into test and training sets
        test_samples = cleaned_samples[:test_size]
        train_samples = cleaned_samples[test_size:]

        print(f"Training set samples: {len(train_samples)}")

        # Save test set
        with open(cleaned_test_file_path, 'w', encoding='utf-8') as fout_test:
            for sample in test_samples:
                fout_test.write(json.dumps(sample, ensure_ascii=False) + '\n')

        # Save training set
        with open(cleaned_train_file_path, 'w', encoding='utf-8') as fout_train:
            for sample in train_samples:
                fout_train.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"Cleaned training set saved to: {cleaned_train_file_path}")

    @staticmethod
    def merge_agent_files(agent1_path: str, agent2_path: str, agent3_path: str, output_path: str) -> None:
        """
        Merge three agent JSONL files into one file.

        Args:
            agent1_path: Path to agent1.jsonl file
            agent2_path: Path to agent2.jsonl file
            agent3_path: Path to agent3.jsonl file
            output_path: Path for the merged output file
        """
        merged_data = []

        # Read agent1.jsonl file content
        with open(agent1_path, 'r', encoding='utf-8') as f1:
            for line in f1:
                merged_data.append(json.loads(line))

        # Read agent2.jsonl file content
        with open(agent2_path, 'r', encoding='utf-8') as f2:
            for line in f2:
                merged_data.append(json.loads(line))

        # Read agent3.jsonl file content
        with open(agent3_path, 'r', encoding='utf-8') as f3:
            for line in f3:
                merged_data.append(json.loads(line))

        # Write to merged file
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for item in merged_data:
                out_file.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Merge complete, file generated: {output_path}")

    @staticmethod
    def check_and_process_files(input_path: str, cleaned_train_path: str,
                               cleaned_test_path: str, test_size: int = 100) -> None:
        """
        Check if cleaned training and test files exist.
        If they don't exist, call process_and_split_jsonl to generate them.

        Args:
            input_path: Path to the input file
            cleaned_train_path: Path for the cleaned training file
            cleaned_test_path: Path for the cleaned test file
            test_size: Number of samples for the test set
        """
        if os.path.exists(cleaned_train_path) and os.path.exists(cleaned_test_path):
            print(f"Files already exist, using: {cleaned_train_path}, {cleaned_test_path}")
        else:
            print(f"Files don't exist, generating: {cleaned_train_path}, {cleaned_test_path}")
            DataProcessor.process_and_split_jsonl(input_path, cleaned_train_path, cleaned_test_path, test_size=test_size)

    @staticmethod
    def create_datasets(tokenizer, args):
        """
        Create training datasets, with control over the number of randomly selected samples via args.num_samples.

        Args:
            tokenizer: Tokenizer to use
            args: Command line arguments

        Returns:
            Dataset: Processed dataset for training
        """
        # Load complete dataset
        dataset = load_dataset(
            "json",
            data_files=args.data_path,
            split="train",
        )

        print(f"Complete training set size: {len(dataset)}")

        # If num_samples is specified, randomly sample
        if args.num_samples != 0:
            num_samples = min(args.num_samples, len(dataset))  # Ensure sample count doesn't exceed dataset size
            sampled_indices = random.sample(range(len(dataset)), num_samples)  # Randomly select indices
            dataset = dataset.select(sampled_indices)  # Select samples by index
            print(f"Training set size after random sampling: {len(dataset)}")
        else:
            print("No sampling specified, using complete dataset.")

        args.nb_examples = len(dataset)

        # Calculate character to token ratio and maximum token count
        chars_per_token, max_token_count = chars_token_ratio(dataset, tokenizer, args)
        print(f"Maximum token count: {max_token_count}")

        # Update maximum sequence length
        args.seq_length = max(max_token_count, args.seq_length)
        print(f"Setting maximum sequence length to: {args.seq_length}")

        # Print character to token ratio
        print(f"Character to token ratio in dataset: {chars_per_token:.2f}")

        # Return processed dataset
        return dataset


class ModelTrainer:
    """
    Class for training language models.
    """

    @staticmethod
    def run_training(args, tokenizer):
        """
        Run model training.

        Args:
            args: Command line arguments
            tokenizer: Tokenizer to use
        """
        print("Loading model")
        print(f"args.seq_length: {args.seq_length}")

        # Set up output directory
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Configure SFT parameters
        sft_config = SFTConfig(
            output_dir=output_dir,
            dataloader_drop_last=True,
            max_seq_length=args.seq_length,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            save_steps=args.save_freq,
            logging_steps=args.log_freq,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.num_warmup_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=args.gradient_checkpointing,
            fp16=args.fp16,
            bf16=args.bf16,
            weight_decay=args.weight_decay,
            run_name=args.run_name,
            report_to="none",
            ddp_find_unused_parameters=False,
            neftune_noise_alpha=5,
            use_liger=True,
        )

        print("Starting main loop")

        # Try to configure quantization with fallback
        try:
            print("Attempting to use 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        except Exception as e:
            print(f"Error setting up BitsAndBytes quantization: {e}")
            print("This could be due to missing Triton or other dependencies.")
            print("Falling back to standard loading without quantization...")
            bnb_config = None

        # Initialize model with Unsloth if specified
        if args.unsloth:
            try:
                # Try to import and use Unsloth
                try:
                    from unsloth import FastLanguageModel
                except (ImportError, ModuleNotFoundError) as e:
                    print(f"Unsloth import failed: {e}")
                    print("Try installing with: pip install unsloth")
                    raise

                print("Using Unsloth acceleration...")

                # Try to load the model with Unsloth
                try:
                    if bnb_config is not None:
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            args.model_path,
                            device_map="auto",
                            quantization_config=bnb_config,
                        )
                    else:
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            args.model_path,
                            device_map="auto",
                        )
                except Exception as e:
                    print(f"Error loading model with Unsloth: {e}")
                    print("This could be due to missing Triton or other dependencies.")
                    raise

                # Try to apply LoRA with Unsloth
                try:
                    model = FastLanguageModel.get_peft_model(
                        model,
                        r=8,
                        target_modules=[
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                        lora_alpha=16,
                        lora_dropout=0.1,  # Dropout = 0 is currently optimized
                        bias="none",  # Bias = "none" is currently optimized
                        use_gradient_checkpointing=True,
                        random_state=3407,
                    )
                except Exception as e:
                    print(f"Error applying LoRA with Unsloth: {e}")
                    raise
            except (ImportError, ModuleNotFoundError) as e:
                print(f"Unsloth import failed: {e}")
                print("Falling back to standard training without Unsloth...")
                args.unsloth = False
                # Initialize model without Unsloth
                try:
                    if bnb_config is not None:
                        model = AutoModelForCausalLM.from_pretrained(
                            args.model_path,
                            device_map="auto",
                            quantization_config=bnb_config,
                        )
                    else:
                        model = AutoModelForCausalLM.from_pretrained(
                            args.model_path,
                            device_map="auto",
                        )
                except Exception as e:
                    print(f"Error loading model with quantization: {e}")
                    print("Trying to load model without quantization...")
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        device_map="auto",
                    )

                # Configure LoRA parameters
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj",
                        "o_proj", "gate_proj",
                        "up_proj", "down_proj",
                    ],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                )

                model = get_peft_model(model, lora_config)
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path,
                )
        else:
            # Initialize model without Unsloth
            print("Using standard training without Unsloth...")
            try:
                if bnb_config is not None:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        device_map="auto",
                        quantization_config=bnb_config,
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        args.model_path,
                        device_map="auto",
                    )
            except Exception as e:
                print(f"Error loading model with quantization: {e}")
                print("Trying to load model without quantization...")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    device_map="auto",
                )

            # Configure LoRA parameters
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=[
                    "q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj",
                    "up_proj", "down_proj",
                ],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, lora_config)
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_path,
            )

        # Load dataset
        train_data = DataProcessor.create_datasets(tokenizer, args)

        # Print training data sample
        print("Training data sample:")
        for i, sample in enumerate(train_data.select(range(1))):
            print(f"Sample {i + 1}: {sample}")

        # Initialize trainer with wandb disabled
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_data,
            tokenizer=tokenizer,  # Explicitly pass tokenizer
        )

        print_trainable_parameters(trainer.model)
        print("Starting training")
        start_time = time.time()

        # Train model
        trainer.train()

        end_time = time.time()
        print("Training complete")
        total_time = end_time - start_time
        print(f"Time taken: {total_time}s ({total_time/60:.2f} minutes)")
        save_path = output_dir
        print(f"Saving model to {save_path}")

        # Save model
        trainer.save_model()


def get_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="CoMaPOI Model Fine-tuning")

    # Model and data path parameters
    parser.add_argument("--model_path", type=str, default="models/", help="Path to pretrained model")
    parser.add_argument("--model", type=str, default="qwen2.5-7b-instruct", help="Pretrained model name")
    parser.add_argument("--dataset", type=str, default="nyc", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="", help="Training data path")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--streaming", action="store_true", help="Use streaming data loading")
    parser.add_argument("--shuffle_buffer", type=int, default=40000, help="Shuffle buffer size")

    # Training parameters
    parser.add_argument("--seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum training steps")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples (0 for all)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay coefficient")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--eos_token_id", type=int, default=49152, help="End of sequence token ID")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--num_warmup_steps", type=int, default=50, help="Warmup steps")
    parser.add_argument("--local_rank", type=int, default=0, help="Local GPU rank")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for data loader")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--log_freq", default=10, type=int, help="Logging frequency")
    parser.add_argument("--save_freq", default=40, type=int, help="Model saving frequency")
    parser.add_argument("--nb_examples", default=20000, type=int, help="Number of examples")
    parser.add_argument("--run_name", default="sft_training", type=str, help="Run name")
    parser.add_argument("--save_name", default="sft_training", type=str, help="Save name")
    parser.add_argument("--agent_id", default=3, type=int, help="Agent ID number")
    parser.add_argument("--type", default='merged', type=str, help="Training type (merged, agent1, agent2, agent3)")
    parser.add_argument("--unsloth", action="store_true", help="Use Unsloth acceleration (single GPU only, not compatible with phi3.5)")
    parser.add_argument('--op_str', type=str, default='4-14', help='Operation string for output directory naming')
    parser.add_argument(
        "--finetune_data_root",
        type=str,
        default=str(FINETUNE_DATA_ROOT),
        help="Root path for fine-tune data",
    )
    parser.add_argument(
        "--finetune_results_root",
        type=str,
        default=str(FINETUNE_RESULTS_ROOT),
        help="Root path for fine-tune outputs",
    )

    return parser.parse_args()


def main():
    """
    Main function to run the fine-tuning process.
    """
    # Ensure wandb is disabled
    os.environ["WANDB_DISABLED"] = "true"

    # Parse arguments
    args = get_args()
    data_root = Path(args.finetune_data_root)
    results_root = Path(args.finetune_results_root)
    dataset_root = data_root / args.dataset

    # Set up file paths for different agent types
    agent1_path = str(dataset_root / "agent1_train_samples.jsonl")
    agent2_path = str(dataset_root / "agent2_train_samples.jsonl")
    agent3_path = str(dataset_root / "agent3_train_samples.jsonl")

    # Process data based on agent type
    if args.type == 'agent1':
        cleaned_train_file_path = str(dataset_root / "agent1_train_samples_all.jsonl")
        cleaned_test_file_path = str(dataset_root / "agent1_train_samples_100.jsonl")
        DataProcessor.check_and_process_files(agent1_path, cleaned_train_file_path, cleaned_test_file_path, test_size=100)
        args.data_path = cleaned_train_file_path

    elif args.type == 'agent2':
        cleaned_train_file_path = str(dataset_root / "agent2_train_samples_all.jsonl")
        cleaned_test_file_path = str(dataset_root / "agent2_train_samples_100.jsonl")
        DataProcessor.check_and_process_files(agent2_path, cleaned_train_file_path, cleaned_test_file_path, test_size=100)
        args.data_path = cleaned_train_file_path

    elif args.type == 'agent3':
        cleaned_train_file_path = str(dataset_root / "agent3_train_samples_all.jsonl")
        cleaned_test_file_path = str(dataset_root / "agent3_train_samples_100.jsonl")
        DataProcessor.check_and_process_files(agent3_path, cleaned_train_file_path, cleaned_test_file_path, test_size=100)
        args.data_path = cleaned_train_file_path

    elif args.type == 'merged':
        merged_file_path = str(dataset_root / "total_agent_train_samples.jsonl")
        # Merge long-term profiles (from agent1), short-term profiles (from agent2), and agent3's reverse inference fine-tuning data
        DataProcessor.merge_agent_files(agent1_path, agent2_path, agent3_path, merged_file_path)

        cleaned_train_file_path = str(dataset_root / "cleaned_total_agent_train_samples_all.jsonl")
        cleaned_test_file_path = str(dataset_root / "cleaned_total_agent_train_samples_100.jsonl")
        DataProcessor.check_and_process_files(merged_file_path, cleaned_train_file_path, cleaned_test_file_path, test_size=100)
        args.data_path = cleaned_train_file_path

    else:
        args.data_path = f"dataset_all/{args.dataset}/train/{args.dataset}_train.jsonl"

    # Set dataset-specific parameters
    args.max_item = DATASET_MAX_ITEM.get(args.dataset, 5091)

    # Set up model path and output directories
    args.model_path = os.path.join(args.model_path, args.model)
    args.run_name = f'bs{args.batch_size}-gas{args.gradient_accumulation_steps}-ms{args.max_steps}-{args.type}-lr{args.learning_rate}'
    args.save_name = f'bs{args.batch_size}-gas{args.gradient_accumulation_steps}-ms{args.max_steps}-{args.type}-lr{args.learning_rate}'
    args.output_dir = str(results_root / args.op_str / f"sft-{args.dataset}" / args.save_name)

    # Print arguments
    print("Parameter list:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Set random seed
    set_seed(args.seed)

    # Set logging level
    logging.set_verbosity_error()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Run training
    ModelTrainer.run_training(args, tokenizer)


if __name__ == "__main__":
    main()
