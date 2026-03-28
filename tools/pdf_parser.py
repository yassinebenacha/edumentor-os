# tools/pdf_parser.py
# PDFParser tool — Extracts text content from student submission PDFs.
#
# Responsibilities:
#   - Accepts a file path or binary content of a PDF document.
#   - Uses PyMuPDF (fitz) to extract raw text, preserving layout where possible.
#   - Handles multi-page documents and returns page-separated text chunks.
#   - Optionally extracts embedded images for future multi-modal grading support.
#   - Cleans and normalizes extracted text (removes artifacts, fixes encoding issues).
#
# Library: pymupdf (fitz)
# Input:  PDF file path (str) or bytes
# Output: List of page text strings or a single concatenated string
# Used by: GraderAgent
