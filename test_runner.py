import os

# Define the values for lora_rank and lora_alpha
lora_rank = [16, 16]
lora_alpha = [8, 8]
batch_size = [16, 32]

lora_tuples = list(zip(lora_rank, lora_alpha, batch_size))

# Base command
base_command = "python3 multitask_classifier.py --use_gpu --fine-tune-mode last-linear-layer --scheduling_mode batch --num_batches_per_epoch 300 --scheduling_policy annealed-sampling --epochs 100 --eval_epochs 10 --lora"

# Run the command with different values
for params in lora_tuples:
    command = f"{base_command} --lora_rank {params[0]} --lora_alpha {params[1]} --batch_size {params[2]}"
    print(f"Running command: {command}")
    os.system(command)