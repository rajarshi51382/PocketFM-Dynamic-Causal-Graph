# Hugging Face Deployment Notes

This guide covers the quick start for deploying to Hugging Face Spaces.

## Quick Setup

1. Go to https://huggingface.co/spaces
2. Click "Create new Space".
3. Name it `causal-character-graphs`.
4. Choose the Streamlit SDK.
5. Upload or clone this repository.

## CLI Option

```
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Create and push to space
huggingface-cli repo create causal-character-graphs --type space --space_sdk streamlit
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/causal-character-graphs
git push hf main
```
