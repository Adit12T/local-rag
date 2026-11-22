import argparse
import logging

import uvicorn

from .config import Config, load_config
from .ollama_client import Ollama
from .server import create_app

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def make_llm(cfg: Config) -> Ollama:
    return Ollama(cfg.ollama_url, cfg.embed_model, cfg.chat_model, cfg.embed_query_prefix, cfg.embed_doc_prefix,
                  cfg.num_ctx, cfg.temperature)


def main():
    parser = argparse.ArgumentParser(description="Local RAG over your allowed folders")
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(args.config)
    if cfg.host not in LOOPBACK and not cfg.auth_token:
        if not cfg.allow_unauthenticated_bind:
            raise SystemExit(f"host is {cfg.host}, which exposes your files to the network. "
                             "Set auth_token in config.yaml, or use host: 127.0.0.1")
        logging.getLogger("rag").warning(
            "Listening on %s without auth_token (allow_unauthenticated_bind). Publish the port "
            "on 127.0.0.1 only, or set auth_token.", cfg.host)
    print(f"\n  Local RAG -> http://{cfg.host}:{cfg.port}\n")
    uvicorn.run(create_app(cfg, make_llm(cfg)), host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
