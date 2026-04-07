FROM python:3.11-slim

# System deps required by Hermes install script
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    xz-utils \
    ripgrep \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash -u 1000 hermes
USER hermes
WORKDIR /home/hermes

# Ensure ~/.local/bin is on PATH (where hermes symlink lands)
ENV PATH="/home/hermes/.local/bin:${PATH}"

# Install Hermes Agent via official script (non-interactive)
RUN curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash -s -- --skip-setup

# Copy custom config and entrypoint
COPY --chown=hermes:hermes config/SOUL.md /home/hermes/.hermes/SOUL.md
COPY --chown=hermes:hermes config/config.yaml /home/hermes/.hermes/config.yaml
COPY --chown=hermes:hermes entrypoint.sh /home/hermes/entrypoint.sh
RUN chmod +x /home/hermes/entrypoint.sh

# Entrypoint writes env vars to ~/.hermes/.env then starts gateway
CMD ["/home/hermes/entrypoint.sh"]
