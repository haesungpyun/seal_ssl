import json
import shutil
import sys

from allennlp.commands import main

config_file = "exp_my/config.json"

# Use overrides to train on CPU. 
# overrides = json.dumps({"trainer.cuda_device": -1, })
overrides="{'trainer.cuda_device':0}"

serialization_dir ="./tmp001"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
if 'tmp' in serialization_dir:
    shutil.rmtree(serialization_dir, ignore_errors=True)


# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "seal",
    "-o", overrides,
    "-f"
]


main()