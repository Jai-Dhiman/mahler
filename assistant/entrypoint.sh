#!/usr/bin/env bash
set -euo pipefail

# Write env vars into Hermes .env file (Hermes reads from ~/.hermes/.env, not process env)
HERMES_ENV="$HOME/.hermes/.env"

{
  echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}"
  echo "DISCORD_BOT_TOKEN=${DISCORD_BOT_TOKEN:-}"
  echo "DISCORD_HOME_CHANNEL=${DISCORD_HOME_CHANNEL:-}"
  echo "DISCORD_ALLOWED_USERS=${DISCORD_ALLOWED_USERS:-}"
} > "$HERMES_ENV"

exec hermes gateway
