import argparse
import gc
import os
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset, load_dataset
from datasets.utils import disable_progress_bar
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizerFast


def _load_model(
    base_dir: str, model_name: str, dtype_str: str
) -> tuple[LlamaForCausalLM, LlamaTokenizerFast]:
    """Load the model from disk."""
    model_path = os.path.join(base_dir, model_name)
    tokenizer = LlamaTokenizerFast.from_pretrained(model_path, local_files_only=True)
    # Pad token is not set by default
    tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[dtype_str],
        device_map="auto",
        local_files_only=True,
    )
    return model, tokenizer


def _load_dataset(base_dir: str, dataset_name: str) -> Dataset:
    """Load the dataset from disk."""
    dataset_path = os.path.join(base_dir, dataset_name)
    dataset = load_dataset(
        "json", data_files=f"{dataset_path}/*.jsonl.zst", keep_in_memory=False
    )
    # dataset = load_from_disk(dataset_path, keep_in_memory=False)
    return dataset["train"]


def _attach_hooks(layers):
    """Attach forward hooks to the decoder layers of the model to extract the outputs."""
    decoder_outputs = defaultdict(list)

    def hook_fn(layer_id):
        def fn(module, input, output):
            # output[0] is of shape (batch_size, seq_len, hidden_dim)
            output_sequence = output[0]
            # convert back to float16 so we can convert to numpy
            if output_sequence.dtype == torch.bfloat16:
                output_sequence = output_sequence.to(torch.float16)
            # Store the entire batch output
            decoder_outputs[layer_id].append(output_sequence.cpu().numpy())

        return fn

    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(hook_fn(i)))

    return hooks, decoder_outputs


def _save_as_dataset(
    output_dir: str,
    decoder_outputs: dict[int, list[torch.Tensor]],
    inputs: list[str],
    chunk_idx: int,
) -> None:
    """Map the raw text inputs to the decoder outputs and save as a dataset."""
    for layer_id, outputs in decoder_outputs.items():
        layer_dir = os.path.join(output_dir, f"layer_{layer_id}")
        os.makedirs(layer_dir, exist_ok=True)

        # Concatenate all outputs across the batch dimension
        # concatenated_outputs = torch.cat(outputs, dim=0)
        concatenated_outputs = np.concatenate(outputs, axis=0)

        dataset_dict = {
            "text_input": inputs,
            "decoder_output": concatenated_outputs,
        }
        dataset = Dataset.from_dict(dataset_dict)

        dataset_path = os.path.join(layer_dir, f"chunk_{chunk_idx}")
        dataset.save_to_disk(dataset_path)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_dir = args.base_dir
    output_dir = args.output_dir or os.path.join(base_dir, "generated_dataset")
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = _load_model(base_dir, args.model_name, args.dtype)

    if args.disable_datasets_progress:
        disable_progress_bar()

    dataset = _load_dataset(base_dir, args.dataset_name)
    hooks, decoder_outputs = _attach_hooks(model.model.layers)

    batch_size = args.batch_size
    buffer_size = args.buffer_size
    max_examples = len(dataset["text"])
    if args.max_examples > 0:
        max_examples = min(max_examples, args.max_examples)

    model.eval()
    with torch.inference_mode():
        inputs_processed = []
        chunk_idx = 0
        for batch_start in tqdm(
            range(0, max_examples, batch_size),
            desc=f"Processing {max_examples} examples in batches",
        ):
            batch_end = min(batch_start + batch_size, max_examples)
            inputs_batch = dataset["text"][batch_start:batch_end]
            inputs_processed.extend(inputs_batch)

            inputs_tokenized = tokenizer(
                inputs_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)
            del inputs_batch

            _ = model(**inputs_tokenized)
            del inputs_tokenized

            # Save at specified buffer_size/frequency, default is every batch
            if len(inputs_processed) >= buffer_size or batch_end >= max_examples:
                _save_as_dataset(
                    output_dir, decoder_outputs, inputs_processed, chunk_idx
                )
                decoder_outputs.clear()
                inputs_processed.clear()
                chunk_idx += 1
            gc.collect()
            torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract decoder outputs from Llama-2-7b-hf"
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.getcwd(),
        help="Base directory where model and dataset are stored (default: ./)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Name of model directory under base_dir",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of dataset directory under base_dir",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the dataset generated from the decoder outputs (default: base_dir/generated_dataset)",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=-1,
        help="Maximum number of examples to process (default: all)",
    )

    parser.add_argument(
        "--buffer-size",
        type=int,
        default=1,
        help="How many examples should be processed and saved in memory before writing to to disk (default: every 1 example)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for batched inference (default: 1)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence tokenization length (default: 1024)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model loading (default: bfloat16)",
    )

    parser.add_argument(
        "--disable-datasets-progress",
        action="store_true",
        help="Disable progress bars from the datasets library",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())