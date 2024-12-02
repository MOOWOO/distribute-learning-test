pip install torch torchvision torchaudio
pip install deepspeed transformers datasets
pip install --upgrade bitsandbytes

sudo chmod -R u+rwx /usr/lib/x86_64-linux-gnu/pdsh
sudo chown root:root /usr/lib

export PDSH_MODULE_DIR=/usr/lib/x86_64-linux-gnu/pdsh

chmod +x run_deepspeed.sh
./run_deepspeed.sh



deepspeed --help
deepspeed --hostfile=hostfile train_llm.py

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