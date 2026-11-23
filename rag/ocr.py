"""Optional OCR for scanned PDF pages (RapidOCR on ONNX Runtime, fully local).

Install with:
    pip install -r requirements-ocr.txt
The recognition models ship inside the package, so nothing is downloaded at run time.
Only pages without a usable text layer are OCR'd.
"""
import logging
import threading
from pathlib import Path

log = logging.getLogger("rag.ocr")


class Ocr:
    def __init__(self, mode: str = "auto"):
        self.enabled = False
        self._engine = None
        self._lock = threading.Lock()
        if mode == "off":
            return
        try:
            import pypdfium2  # noqa: F401
            import rapidocr_onnxruntime  # noqa: F401

            self.enabled = True
        except ImportError:
            if mode == "on":
                raise SystemExit("ocr: on, but the OCR packages are not installed. "
                                 "Run: pip install -r requirements-ocr.txt")
            log.info("OCR not installed (optional): pip install -r requirements-ocr.txt")

    def pdf_pages(self, path: Path, page_indexes: list[int]) -> dict[int, str]:
        """Recognised text for the given 0-based pages of a PDF."""
        if not self.enabled or not page_indexes:
            return {}
        import numpy as np
        import pypdfium2 as pdfium
        from rapidocr_onnxruntime import RapidOCR

        out = {}
        with self._lock:
            if self._engine is None:
                self._engine = RapidOCR()
            pdf = pdfium.PdfDocument(str(path))
            try:
                for i in page_indexes:
                    image = pdf[i].render(scale=2).to_pil().convert("RGB")
                    result, _ = self._engine(np.array(image))
                    out[i] = "\n".join(line[1] for line in result or [])
            finally:
                pdf.close()
        return out
