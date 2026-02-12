FROM python:3.11-slim

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        python3-tk \
        tk \
        tcl \
        xvfb \
        xauth \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . ./

RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir .

CMD ["xvfb-run", "-a", "python", "-m", "uao_neumonia"]
