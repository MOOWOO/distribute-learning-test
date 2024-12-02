pip install torch torchvision torchaudio
pip install deepspeed transformers datasets

deepspeed --help
deepspeed --hostfile=hostfile train_llm.py
