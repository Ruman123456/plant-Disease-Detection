#!/usr/bin/env bash
# exit on error
set -o errexit

# Install python dependencies
pip install -r requirement.txt

# Re-assemble the split TFLite model chunks into the full model file
echo "Reassembling the TFLite model..."
cat model.tflite.part* > model.tflite
echo "Model reassembled successfully."
