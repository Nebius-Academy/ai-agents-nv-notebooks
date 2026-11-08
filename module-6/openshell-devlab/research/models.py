"""Per-role LLM routing on Nebius Token Factory.
"""

import os

from langchain_nebius import ChatNebius

LEAD_MODEL_NAME = os.environ.get("LEAD_MODEL", "MiniMaxAI/MiniMax-M3")
WORKER_MODEL_NAME = os.environ.get("WORKER_MODEL", "MiniMaxAI/MiniMax-M3")
# Keep the client endpoint aligned with the OpenShell policy and provider profile.
REGION = os.environ.get("TOKEN_FACTORY_REGION", "us-central1")
BASE_URL = f"https://api.tokenfactory.{REGION}.nebius.com/v1/"

# The sandbox receives the credential as TOFA_API_KEY; ChatNebius would otherwise
# look for NEBIUS_API_KEY.
API_KEY = os.environ["TOFA_API_KEY"]

LEAD_MODEL = ChatNebius(
    model=LEAD_MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    reasoning_effort="high",
)

WORKER_MODEL = ChatNebius(
    model=WORKER_MODEL_NAME,
    api_key=API_KEY,
    base_url=BASE_URL,
    reasoning_effort="medium",
)
