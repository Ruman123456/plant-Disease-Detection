#!/usr/bin/env bash
# exit on error
set -o errexit

# Install python dependencies
pip install -r requirements.txt

# Re-assemble the split TFLite model chunks into the full model file
echo "Reassembling the TFLite model..."
cat model_data/model.tflite.part* > model_data/model.tflite
echo "Model reassembled successfully."
