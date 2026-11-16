"""Watch the allowed paths and keep the index current automatically.

File-system events are debounced (editors fire several events per save), then
applied to the index. A periodic full rescan is a safety net for missed events.
"""
import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .index import Indexer

log = logging.getLogger("rag.watcher")


class _Handler(FileSystemEventHandler):
    def __init__(self, watcher: "Watcher"):
        self.watcher = watcher

    def on_any_event(self, event):
        if event.event_type in ("opened", "closed_no_write"):
            return
        if event.is_directory and event.event_type == "modified":
            return  # fires for every child change; the child's own event is enough
        self.watcher.touch(event.src_path)
        dest = getattr(event, "dest_path", "")
        if dest:
            self.watcher.touch(dest)


class Watcher:
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.cfg = indexer.cfg
        self.pending: dict[str, float] = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.observer = Observer()

    def touch(self, path):
        if isinstance(path, bytes):
            path = path.decode(errors="ignore")
        if any(part in self.cfg.ignore for part in Path(path).parts):
            return
        with self.lock:
            self.pending[path] = time.time()

    def start(self):
        handler = _Handler(self)
        watched = set()
        for p in self.cfg.paths:
            target, recursive = (p, True) if p.is_dir() else (p.parent, False)
            if (target, recursive) not in watched:
                self.observer.schedule(handler, str(target), recursive=recursive)
                watched.add((target, recursive))
        self.observer.start()
        threading.Thread(target=self._drain, daemon=True, name="rag-watch").start()
        threading.Thread(target=self._rescan, daemon=True, name="rag-rescan").start()
        log.info("Watching %d path(s) for changes", len(self.cfg.paths))

    def stop(self):
        self.stop_event.set()
        self.observer.stop()
        self.observer.join(timeout=5)

    def _drain(self):
        while not self.stop_event.wait(0.5):
            cutoff = time.time() - self.cfg.debounce_seconds
            with self.lock:
                ready = [p for p, t in self.pending.items() if t <= cutoff]
                for p in ready:
                    del self.pending[p]
            for p in ready:
                try:
                    self._apply(Path(p))
                except Exception:
                    log.exception("Failed to process change for %s", p)

    def _apply(self, path: Path):
        if path.is_dir():
            self.indexer.sync(scope=path)      # new or moved-in folder
        elif path.exists():
            self.indexer.index_file(path)      # created or modified file
        else:
            self.indexer.remove_path(path)     # deleted file or folder

    def _rescan(self):
        interval = self.cfg.rescan_minutes * 60
        if interval <= 0:
            return
        while not self.stop_event.wait(interval):
            try:
                stats = self.indexer.sync()
                log.info("Periodic rescan: %s", stats)
            except Exception:
                log.exception("Periodic rescan failed")
