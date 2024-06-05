import os

'''
parser.add_argument("--parallel_adaption_layers", type=str, default="", choices=('low-rank', 'pals', 'pals-shared', 'mixed'), help="Use parallel adaption layers for BERT")
parser.add_argument("--adaption_layer_late_attach", action='store_true', help="Attach adaption layers at end of BERT layers")
parser.add_argument("--adaption_layer_shared_attention", action='store_true', help="Use shared attention for adaption layers")
'''


# TEST 1
pal_type = [
        "low-rank",
        "low-rank",
        "pals",
        "pals",
        "pals-shared",
        "pals-shared",
        "mixed",
        "mixed"
    ]

shared_attention = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False
    ]

late_attach = [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
        True
    ]

#test_tuples = list(zip(pal_type, late_attach, shared_attention))

# TEST 2
pal_type = [
        "pals-shared",
        "pals-shared"
    ]

shared_attention = [
        True,
        True
    ]

late_attach = [
        True,
        False
    ]

test_tuples = list(zip(pal_type, late_attach, shared_attention))

# Base command
base_command = "python3 multitask_classifier.py --use_gpu --epochs 100 --batch_size 16 --fine-tune-mode full-model --scheduling_mode batch --scheduling_policy annealed-sampling --num_batches_per_epoch 300"

# Run the command with different values
for params in test_tuples:
    command = f"{base_command} --parallel_adaption_layers {params[0]} {'--adaption_layer_late_attach ' if params[1] else ''} {'--adaption_layer_shared_attention ' if params[2] else ''}"
    print(f"Running command: {command}")
    os.system(command)