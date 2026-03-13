# Deployment Guide

This document explains how to deploy the demo on multiple platforms.

## Streamlit Community Cloud (Recommended)

Result: `https://your-app-name.streamlit.app`

1. Go to https://share.streamlit.io
2. Sign in with GitHub.
3. Click "New app".
4. Select this repository: `itsloganmann/PocketFM-Dynamic-Causal-Graph-MVP`.
5. Set:
   - Branch: `production`
   - Main file path: `streamlit_app.py`
6. In Advanced settings, set a custom subdomain if desired.
7. Click Deploy.

Your app will be live at `https://causal-character-graphs.streamlit.app` within a few minutes.

### CI/CD Notes (GitHub Actions to Streamlit Cloud)

- Tests run on pushes and PRs to `main`, and on `production`.
- Streamlit Cloud deploys from the `production` branch only.
- Promote changes via the GitHub Actions workflow in `.github/workflows/promote.yml`.
- Protect `production` so only passing CI can be promoted.

## Hugging Face Spaces

Result: `https://huggingface.co/spaces/your-username/causal-character-graphs`

1. Go to https://huggingface.co/spaces
2. Click "Create new Space".
3. Name it `causal-character-graphs`.
4. Select "Streamlit" as the SDK.
5. Clone or upload this repository's files to the space.

CLI option:

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

## Render

Result: `https://causal-character-graphs.onrender.com`

1. Go to https://render.com
2. Create a free account.
3. Click "New Web Service".
4. Connect your GitHub repository.
5. Set:
   - Name: `causal-character-graphs`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
6. Deploy.

## Features That Work Without API Keys

- Belief revision: log-odds updates as you chat
- Causal propagation: belief graph with A to B dependencies
- Emotional state: valence and arousal tracking
- Relationship tracking: trust, affection, respect per entity
- Smart dialogue: rule-based contextually-aware character responses
- Scenario presets: one-click narrative events
- Save, load, download: persist and restore the full simulation state

## Optional: Enhanced LLM Responses

To enable LLM-powered responses (optional):
1. Get a Gemini API key from https://ai.google.dev
2. Enter it in the sidebar of the app

The app works without this; the rule-based system generates responses by default.
