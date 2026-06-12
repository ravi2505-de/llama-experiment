import gc
import os
import time

import gradio as gr
import psutil
from llama_cpp import Llama


MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

N_CTX = int(os.getenv("N_CTX", "2048"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

PROCESS = psutil.Process(os.getpid())
LLM = None


def memory_mb() -> float:
    gc.collect()
    return PROCESS.memory_info().rss / (1024 * 1024)


def log_memory(label: str) -> None:
    print(f"[memory] {label}: {memory_mb():.1f} MB", flush=True)


def load_model() -> Llama:
    global LLM
    if LLM is not None:
        return LLM

    print(f"[startup] repo={MODEL_REPO}", flush=True)
    print(f"[startup] file={MODEL_FILE}", flush=True)
    print("[startup] loading model with CPU-only llama.cpp settings", flush=True)
    log_memory("before_model_load")
    started = time.time()

    LLM = Llama.from_pretrained(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        n_ctx=N_CTX,
        n_threads=max(1, os.cpu_count() or 1),
        n_gpu_layers=0,
        verbose=True,
    )

    print(f"[startup] model_loaded_seconds={time.time() - started:.2f}", flush=True)
    log_memory("after_model_load")
    return LLM


def generate_response(message: str, history: list[dict]) -> str:
    del history

    message = (message or "").strip()
    if not message:
        return "Send a prompt to verify generation."

    llm = load_model()

    log_memory("before_generation")
    started = time.time()
    result = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a concise assistant running in a minimal CPU-only Hugging Face Space.",
            },
            {"role": "user", "content": message},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=False,
    )
    elapsed = time.time() - started
    log_memory("after_generation")
    print(f"[generation] seconds={elapsed:.2f}", flush=True)

    return result["choices"][0]["message"]["content"].strip()


with gr.Blocks(title="Minimal Qwen llama-cpp CPU") as demo:
    gr.Markdown("# Minimal Qwen llama-cpp CPU")
    gr.Markdown(
        "Qwen2.5-1.5B-Instruct GGUF Q4_K_M, CPU-only, non-streaming. "
        "Startup and generation memory are printed in the Space logs."
    )
    gr.ChatInterface(
        fn=generate_response,
        chatbot=gr.Chatbot(type="messages", height=420),
        textbox=gr.Textbox(placeholder="Ask a short test question", scale=7),
        type="messages",
        examples=[
            "Say hello in one sentence.",
            "What is 2 + 2?",
        ],
    )


load_model()

if __name__ == "__main__":
    demo.launch()
