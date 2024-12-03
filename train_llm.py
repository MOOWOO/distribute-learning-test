import os
import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Configuration
MODEL_NAME = "EleutherAI/pythia-6.9b"
SEQ_LEN = 256  # Reduced sequence length
BATCH_SIZE_PER_GPU = 1
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 3
LR = 3e-5

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))  # Total number of GPUs across nodes
TRAIN_BATCH_SIZE = BATCH_SIZE_PER_GPU * WORLD_SIZE * GRADIENT_ACCUMULATION_STEPS

def get_dataloader(tokenizer):
    """Prepare the DataLoader."""
    # Training Loss 7.4112
    # Epoch 3.0
    # Step 1692
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=SEQ_LEN)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return torch.utils.data.DataLoader(tokenized_dataset, batch_size=BATCH_SIZE_PER_GPU, shuffle=True)

def train():
    print(f"[INFO] Starting training on {WORLD_SIZE} GPUs")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add a pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Resize model embeddings to account for added special tokens
    model.resize_token_embeddings(len(tokenizer))

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params={
            "train_batch_size": TRAIN_BATCH_SIZE,
            "micro_batch_size_per_gpu": BATCH_SIZE_PER_GPU,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": LR, "betas": [0.9, 0.999], "eps": 1e-8}
            },
            "fp16": {"enabled": True},
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True}
            }
        }
    )

    # Load DataLoader
    dataloader = get_dataloader(tokenizer)

    # Training loop
    for epoch in range(EPOCHS):
        print(f"[INFO] Epoch {epoch + 1}/{EPOCHS}")
        model_engine.train()
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].to(model_engine.local_rank)

            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0 and model_engine.local_rank == 0:
                print(f"[LOG] Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

    print("[INFO] Training completed.")

if __name__ == "__main__":
    train()
