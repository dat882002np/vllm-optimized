FROM vllm/vllm-openai:v0.26.0

# Copy our modified Python files to overwrite the original ones in site-packages
COPY vllm/v1/attention/backends/flashinfer.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py
COPY vllm/v1/core/block_pool.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py
COPY vllm/v1/core/sched/scheduler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
