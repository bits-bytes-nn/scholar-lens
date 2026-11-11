"""Save a streamed PDF response with a size cap and a magic-byte check.

Shared by arXiv downloads and arbitrary PDF URLs so both enforce the same cap
and never leave a partial file behind.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Cap PDF downloads so a hostile or oversized URL cannot exhaust disk or memory.
MAX_PDF_BYTES = 100 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
# Readers accept the header anywhere in the first KiB (after a BOM or junk).
MAGIC_WINDOW = 1024
CHUNK_BYTES = 1 << 16


class PdfTooLargeError(ValueError):
    """The PDF is larger than :data:`MAX_PDF_BYTES`."""


class NotPdfContentError(ValueError):
    """The body is not a PDF (an HTML landing page, an error page …)."""


def save_pdf(
    chunks: Iterable[bytes],
    path: Path,
    *,
    source: str,
    declared_length: str | None = None,
) -> Path:
    """Write ``chunks`` to ``path``. Raises :class:`PdfTooLargeError` past the
    cap (checked against ``declared_length`` first, then while streaming) and
    :class:`NotPdfContentError` when no PDF header appears in the first
    :data:`MAGIC_WINDOW` bytes; on either error no file is left."""
    cap = MAX_PDF_BYTES
    if declared_length and declared_length.isdigit() and int(declared_length) > cap:
        raise PdfTooLargeError(
            f"PDF at '{source}' exceeds the {cap}-byte limit "
            f"(Content-Length: {declared_length})."
        )
    written = 0
    head = b""
    try:
        with path.open("wb") as fh:
            for chunk in chunks:
                if not chunk:
                    continue
                written += len(chunk)
                if written > cap:
                    raise PdfTooLargeError(
                        f"PDF at '{source}' exceeds the {cap}-byte limit."
                    )
                if len(head) < MAGIC_WINDOW:
                    head += chunk[: MAGIC_WINDOW - len(head)]
                    if len(head) == MAGIC_WINDOW and PDF_MAGIC not in head:
                        raise NotPdfContentError(
                            f"Content from '{source}' is not a PDF."
                        )
                fh.write(chunk)
        if PDF_MAGIC not in head:
            raise NotPdfContentError(f"Content from '{source}' is not a PDF.")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path
