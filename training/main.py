from functools import partial

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizerFast,
    Trainer,
    TrainingArguments,
)


class LlamaLayerTrainer(nn.Module):
    def __init__(self, model_path: str, target_layer: int):
        super().__init__()
        # Load the full model
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            output_hidden_states=True,  # Ensure hidden states are stored in memory for loss calculation
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = LlamaTokenizerFast.from_pretrained(model_path)
        # Llama tokenizer does not set a pad token by default
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.target_layer = target_layer
        target = self.model.model.layers[target_layer]

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze target layer
        for param in target.parameters():
            param.requires_grad = True

        # Zero out target layer weights
        def zero_weights(module):
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.zero_()
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        target.apply(zero_weights)

    def forward(self, input_ids, attention_mask, **kwargs):
        # We return the full output object for simplicity
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    # Fixes some issues with Trainer with gradient checkpointing.
    @property
    def config(self):
        return self.model.config

    @property
    def device(self):
        return self.model.device

    @property
    def main_input_name(self):
        return "input_ids"


def collate_fn(
    examples: list[dict], tokenizer: LlamaTokenizerFast
) -> dict[str, torch.Tensor]:
    texts = [ex["text_input"] for ex in examples]

    # Note that we must tokenize in the same manner as the dataset generation
    tokenized_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )

    target_hidden_states_list = []
    # Max sequence length in the batch
    max_len = tokenized_inputs["input_ids"].shape[1]
    for ex in examples:
        # Convert hidden states to tensor
        hs = torch.tensor(ex["decoder_output"], dtype=torch.float32)

        # Data generation was not batched, but our training will be done in batches.
        # This means that some of the decoder outputs can be shorter than required since our tokenizer is not working with a batched input.
        # I.e., we must pad (we include truncation here for generalization) the hidden states to the max length of the batch
        current_len = hs.shape[0]
        if current_len > max_len:
            # This case should not happen if max_length is consistent with how we generated the dataset
            hs = hs[:max_len, :]
            assert False, "Something is wrong with the dataset"
        elif current_len < max_len:
            padding_size = max_len - current_len
            # Ensure padding is on the same device as hidden states
            padding = torch.zeros((padding_size, hs.shape[1]), dtype=hs.dtype)
            hs = torch.cat([hs, padding], dim=0)

        target_hidden_states_list.append(hs)

    # Stack hidden states, shape is (batch_size, seq_len, hidden_dim=4096)
    target_hidden_states = torch.stack(target_hidden_states_list)

    # Note we need the attention mask for the loss calculation
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "target_hidden_states": target_hidden_states,
    }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # Ensure target_hidden_states is on the correct device
        target_hidden_states = inputs["target_hidden_states"].to(
            self.accelerator.device
        )

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # Extract the hidden states from the target layer
        # outputs.hidden_states is a tuple: (embeddings, layer_0_output, layer_1_output, ...)
        actual_hidden_states = outputs.hidden_states[model.target_layer + 1]

        target_hidden_states = target_hidden_states.to(actual_hidden_states.dtype)

        loss_fct = nn.MSELoss(reduction="none")
        loss = loss_fct(actual_hidden_states, target_hidden_states)

        # Loss shape: (batch_size, seq_len, hidden_dim)
        # Attention mask shape: (batch_size, seq_len)
        mask = attention_mask.unsqueeze(-1).expand_as(loss)
        typed_mask = mask.to(loss.dtype)

        masked_loss = loss * typed_mask
        # We average over the non masked tokens (not padding)
        mean_loss = masked_loss.sum().float() / typed_mask.sum().float()

        return (mean_loss, outputs) if return_outputs else mean_loss


def main():
    # model_path = os.path.join(os.getenv("SLURM_TMPDIR", ""), "Llama-2-7b-hf")
    # dataset_path = os.path.join(os.getenv("SLURM_TMPDIR", ""), "layer_1")
    model_path = "Llama-2-7b-hf"
    dataset_path = "layer_1"
    target_layer = 1 # (0 to 31)

    # Create a Llama wrapper that zeroes out target layer, among other things
    model = LlamaLayerTrainer(model_path, target_layer)
    dataset = load_from_disk(dataset_path)

    split = dataset.train_test_split(test_size=0.1, seed=137)
    train_dataset = split["train"]
    val_dataset = split["test"]

    training_args = TrainingArguments(
        output_dir=f"./llama_layer_{target_layer}_train",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch = 2*8=16
        num_train_epochs=5,
        logging_dir="./logs",
        logging_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        report_to="tensorboard",
        learning_rate=2e-5,
        bf16=True,  # Use bf16 for mixed precision training
        gradient_checkpointing=True,  # Massively reduce GPU memory usage which we need for this script
        gradient_checkpointing_kwargs={"use_reentrant": False},
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        remove_unused_columns=False,  # Ensure columns are kept in dataset as we use a custom data processor (collate_fn)
    )

    # Create partial function for collate_fn with the llama tokenizer
    collate_fn_with_tokenizer = partial(collate_fn, tokenizer=model.tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_with_tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    print("Saving best model...")
    trainer.save_model(f"./llama_layer_{target_layer}_best")
    print("Model saved.")


if __name__ == "__main__":
    main()
