import subprocess
import pandas as pd

# The combinations you want to test
models = ['resnet', 'densenet', 'efficientnet']
levels = [0, 1, 2, 3, 4]

# This script calls your working script 15 times
for m in models:
    for l in levels:
        print(f"--- Benchmarking {m} at unfreeze level {l} ---")
        try:
            subprocess.run([
                'python', '03_modeling/03_train_phase2_vision_only.py', 
                '--model', m, 
                '--unfreeze_blocks', str(l)
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"Skipping {m} level {l} due to training error.")