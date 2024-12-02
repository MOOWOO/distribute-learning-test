pip uninstall -y torch torchvision torchaudio flash-attn

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip uninstall transformer-engine bitsandbytes

pip install deepspeed transformers datasets

pip install --upgrade bitsandbytes

pip install --upgrade transformers

sudo chmod -R u+rwx /usr/lib/x86_64-linux-gnu/pdsh

sudo chown root:root /usr/lib

export PDSH_MODULE_DIR=/usr/lib/x86_64-linux-gnu/pdsh

deepspeed --help

deepspeed --hostfile hostfile --master_port=29501 train_llm.py