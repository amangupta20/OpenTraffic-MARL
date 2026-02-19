FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SUMO_HOME=/usr/share/sumo

# Install system deps + SUMO
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3 python3-pip python3-dev \
    wget gnupg2 ca-certificates \
    # noVNC / visual demo deps
    xvfb x11vnc novnc websockify \
    && add-apt-repository ppa:sumo/stable \
    && apt-get update \
    && apt-get install -y --no-install-recommends sumo sumo-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source & SUMO net
COPY src/ src/
COPY sumo_net/ sumo_net/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create dirs for model + TB logs
RUN mkdir -p models tb_logs

# Prometheus metrics port
EXPOSE 8000
# noVNC port (for demo mode)
EXPOSE 6080

ENV MODE=dumb
ENTRYPOINT ["/entrypoint.sh"]
