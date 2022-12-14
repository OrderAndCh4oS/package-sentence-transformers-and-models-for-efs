import subprocess
import sys

sentence_transformers_path = '/Users/seancooper/code/sentence_transformers/sentence_transformers_lib'

subprocess.check_call(
        [sys.executable, "-m", "pip", "install", 'sentence_transformers', f'--target={sentence_transformers_path}']
    )
