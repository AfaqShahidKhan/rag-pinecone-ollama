"""
src/infrastructure/validation/document_protection_validator.py

Checks whether a document has been marked read-only through the file
FORMAT's own native protection mechanism — not the OS filesystem
permission bit. Matches how a document actually gets marked "finished, do
not edit" in practice:
  .docx  -> Word's Protect Document > Always Open Read-Only / Restrict Editing
  .pptx  -> PowerPoint's Protect Presentation > Always Open Read-Only / password
  .pdf   -> owner-password content-modification restriction

Only formats with a native protection mechanism are checked. A file is
rejected if that mechanism exists but was never turned on. Formats with
no native protection concept (json, html, images, and legacy .ppt — a
binary OLE2 format, not inspectable the same way) always pass — there's
nothing to check, so nothing to enforce.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from pypdf import PdfReader
from pypdf.constants import UserAccessPermissions

from src.domain.errors import ErrorCode, PipelineError
from src.domain.interfaces import IFileValidator

_WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_SLIDE_NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


class DocumentProtectionValidator(IFileValidator):
    def validate(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".docx":
            self._validate_docx(path)
        elif suffix == ".pptx":
            self._validate_pptx(path)
        elif suffix == ".pdf":
            self._validate_pdf(path)
        # every other format (including legacy .ppt): no native protection
        # concept we can cheaply check — always passes

    def _validate_docx(self, path: Path) -> None:
        if self._is_docx_protected(path):
            return
        raise PipelineError(
            ErrorCode.FILE_NOT_READONLY,
            "DOCX is not protected — no 'Always Open Read-Only' or 'Restrict "
            "Editing' set (Word: File > Info > Protect Document).",
            file_path=str(path),
        )

    def _validate_pptx(self, path: Path) -> None:
        if self._is_pptx_protected(path):
            return
        raise PipelineError(
            ErrorCode.FILE_NOT_READONLY,
            "PPTX is not protected — no 'Always Open Read-Only' or password "
            "protection set (PowerPoint: File > Info > Protect Presentation).",
            file_path=str(path),
        )

    def _validate_pdf(self, path: Path) -> None:
        if self._is_pdf_protected(path):
            return
        raise PipelineError(
            ErrorCode.FILE_NOT_READONLY,
            "PDF has no content-modification restriction set "
            "(no owner-password permissions restricting editing).",
            file_path=str(path),
        )

    @staticmethod
    def _is_docx_protected(path: Path) -> bool:
        try:
            with zipfile.ZipFile(path) as z:
                with z.open("word/settings.xml") as f:
                    root = ET.parse(f).getroot()
        except (KeyError, zipfile.BadZipFile, ET.ParseError):
            return False  # unreadable/malformed — content validation catches this later

        if root.find("w:writeProtection", _WORD_NS) is not None:
            return True  # "Always Open Read-Only"

        protection = root.find("w:documentProtection", _WORD_NS)
        if protection is not None:
            enforcement = protection.get(f"{{{_WORD_NS['w']}}}enforcement")
            if enforcement in ("1", "true", "on"):
                return True  # "Restrict Editing", actively enforced

        return False

    @staticmethod
    def _is_pptx_protected(path: Path) -> bool:
        try:
            with zipfile.ZipFile(path) as z:
                with z.open("ppt/presentation.xml") as f:
                    root = ET.parse(f).getroot()
        except (KeyError, zipfile.BadZipFile, ET.ParseError):
            return False

        # PowerPoint uses one element for both "Always Open Read-Only" and
        # password protection — presence means protected either way.
        return root.find("p:modifyVerifier", _SLIDE_NS) is not None

    @staticmethod
    def _is_pdf_protected(path: Path) -> bool:
        try:
            reader = PdfReader(str(path))
            if not reader.is_encrypted:
                return False
            try:
                reader.decrypt("")
            except Exception:
                pass
            perms = reader.user_access_permissions
            if perms is None:
                return False
            return not bool(perms & UserAccessPermissions.MODIFY)
        except Exception:
            return False  # unreadable — content validation catches this later