# Welcome to Local RAG in Docker

This folder is what the container indexes when you have not mounted your own.

To chat with your own files, set `RAG_DOCS` to a folder before starting:

    RAG_DOCS=~/Documents/notes docker compose up -d

The folder is mounted read-only at `/docs`. Local RAG never writes to it.
