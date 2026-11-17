"""Save a streamed PDF response with a size cap and a magic-byte check.

Shared by arXiv downloads and arbitrary PDF URLs so both enforce the same cap
and never leave a partial file behind.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

# Cap PDF downloads so a hostile or oversized URL cannot exhaust disk or memory.
MAX_PDF_BYTES = 100 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
# Readers accept the header anywhere in the first KiB (after a BOM or junk).
MAGIC_WINDOW = 1024
CHUNK_BYTES = 1 << 16
# The whole download's deadline: the read timeout is per chunk, so a server
# that trickles a chunk every few seconds would otherwise hold the job until
# its Batch timeout. 100 MB at ~0.5 MB/s still fits.
PDF_DOWNLOAD_DEADLINE_SECONDS = 300.0


class PdfTooLargeError(ValueError):
    """The PDF is larger than :data:`MAX_PDF_BYTES`."""


class NotPdfContentError(ValueError):
    """The body is not a PDF (an HTML landing page, an error page …)."""


class PdfDownloadTooSlowError(TimeoutError):
    """The download did not finish within its deadline."""


def save_pdf(
    chunks: Iterable[bytes],
    path: Path,
    *,
    source: str,
    declared_length: str | None = None,
    deadline_seconds: float = PDF_DOWNLOAD_DEADLINE_SECONDS,
) -> Path:
    """Write ``chunks`` to ``path``. Raises :class:`PdfTooLargeError` past the
    cap (checked against ``declared_length`` first, then while streaming) and
    :class:`NotPdfContentError` when no PDF header appears in the first
    :data:`MAGIC_WINDOW` bytes, and :class:`PdfDownloadTooSlowError` when
    the whole download takes over ``deadline_seconds``; on any error no
    file is left."""
    cap = MAX_PDF_BYTES
    if declared_length and declared_length.isdigit() and int(declared_length) > cap:
        raise PdfTooLargeError(
            f"PDF at '{source}' exceeds the {cap}-byte limit "
            f"(Content-Length: {declared_length})."
        )
    deadline = _clock() + deadline_seconds
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
                if _clock() > deadline:
                    raise PdfDownloadTooSlowError(
                        f"PDF at '{source}' took over {deadline_seconds:g} s."
                    )
        if PDF_MAGIC not in head:
            raise NotPdfContentError(f"Content from '{source}' is not a PDF.")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


_clock = time.monotonic  # replaced in tests
