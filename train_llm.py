import torch
import deepspeed
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델 경로와 하이퍼파라미터
MODEL_NAME = "meta-llama/Llama-3.1-8b"  # 사용할 LLaMA 3.1 모델
BATCH_SIZE = 16
SEQ_LEN = 512
EPOCHS = 3
LR = 3e-5

def get_data_loader():
    """
    Wikitext-2-raw 데이터셋을 로드하고 토크나이징하는 함수.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 데이터 로드 및 토크나이징
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=SEQ_LEN)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return torch.utils.data.DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=True)

def train():
    # DeepSpeed 초기화
    deepspeed.init_distributed()

    # 모델 및 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 모델을 DeepSpeed로 래핑
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params={
            "train_batch_size": BATCH_SIZE,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": LR,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True  # FP16 활성화
            },
        }
    )

    # 데이터 로더 생성
    train_loader = get_data_loader()

    # 학습 루프
    for epoch in range(EPOCHS):
        model_engine.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].to(model_engine.local_rank)

            # 모델 실행
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # 역전파 및 매개변수 업데이트
            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0 and model_engine.local_rank == 0:
                print(f"Epoch {epoch + 1}/{EPOCHS}, Step {step}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
