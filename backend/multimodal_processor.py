"""
Multimodal processor for BNP Paribas Cardif Claims Management.
Handles OCR document processing, image analysis, and vision-based damage assessment.
"""

import io
import logging
import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from backend.config import settings, logger


class MultimodalProcessor:
    """
    Processes uploaded documents: PDFs, images (JPG/PNG), and CSV data.
    Supports OCR text extraction and vision-based damage assessment.
    """

    def __init__(self):
        """Initialize the multimodal processor."""
        self._tesseract_available = None
        self._pdfplumber_available = None
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check which optional dependencies are available."""
        try:
            import pytesseract
            if settings.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info("Tesseract OCR is available.")
        except (ImportError, Exception) as e:
            self._tesseract_available = False
            logger.warning(f"Tesseract OCR not available: {e}")

        try:
            import pdfplumber
            self._pdfplumber_available = True
            logger.info("pdfplumber is available.")
        except ImportError:
            self._pdfplumber_available = False
            logger.warning("pdfplumber not available. PDF processing limited.")

    def process_document(self, file_path: str, mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document file and extract information.

        Args:
            file_path: Path to the document file.
            mime_type: MIME type of the document. If None, inferred from extension.

        Returns:
            Dictionary with extracted text, structured data, and metadata.
        """
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}", "success": False}

        mime = mime_type or self._infer_mime_type(path)
        logger.info(f"Processing document: {path.name} (type: {mime})")

        result = {
            "filename": path.name,
            "file_path": str(path.resolve()),
            "mime_type": mime,
            "success": True,
            "ocr_text": "",
            "extracted_data": {},
            "vision_description": "",
        }

        try:
            if mime.startswith("image/"):
                img_result = self._process_image(str(path))
                result.update(img_result)
            elif mime == "application/pdf":
                pdf_result = self._process_pdf(str(path))
                result.update(pdf_result)
            elif mime in ("text/csv", "text/plain"):
                text_result = self._process_text(str(path))
                result.update(text_result)
            else:
                # Try as image or text fallback
                try:
                    img_result = self._process_image(str(path))
                    result.update(img_result)
                except Exception:
                    text_result = self._process_text(str(path))
                    result.update(text_result)

        except Exception as e:
            logger.error(f"Error processing document {path.name}: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    def _infer_mime_type(self, path: Path) -> str:
        """Infer MIME type from file extension."""
        ext = path.suffix.lower()
        mime_map = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".json": "application/json",
        }
        return mime_map.get(ext, "application/octet-stream")

    def _process_image(self, file_path: str) -> Dict[str, Any]:
        """
        Process an image file: OCR + vision description.

        Returns dict with ocr_text and vision_description.
        """
        image = Image.open(file_path)
        result = {}

        # OCR text extraction
        text = ""
        if self._tesseract_available:
            try:
                import pytesseract
                text = pytesseract.image_to_string(image, lang="eng+fra")
                text = text.strip()
                logger.info(f"OCR extracted {len(text)} characters from image.")
            except Exception as e:
                logger.warning(f"OCR failed for image: {e}")
                text = "[OCR extraction failed]"
        else:
            text = "[OCR not available - Tesseract not installed]"

        result["ocr_text"] = text

        # Vision description (damage assessment)
        vision_desc = self._analyze_image_damage(image, text)
        result["vision_description"] = vision_desc

        # Extract structured data from OCR text
        result["extracted_data"] = self._extract_structured_data(text)

        return result

    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file: extract text with pdfplumber.

        Returns dict with ocr_text.
        """
        text_parts = []

        if self._pdfplumber_available:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text() or ""
                        text_parts.append(f"--- Page {i + 1} ---\n{page_text}")
                logger.info(f"PDF extracted {sum(len(t) for t in text_parts)} characters from {len(text_parts)} pages.")
            except Exception as e:
                logger.warning(f"PDF extraction failed: {e}")
                text_parts.append("[PDF extraction failed]")
        else:
            text_parts.append("[PDF extraction not available - pdfplumber not installed]")

        text = "\n\n".join(text_parts)

        return {
            "ocr_text": text,
            "extracted_data": self._extract_structured_data(text),
            "vision_description": "",
        }

    def _process_text(self, file_path: str) -> Dict[str, Any]:
        """Process a text/CSV file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            extracted = {}
            if file_path.endswith(".csv"):
                extracted["format"] = "csv"
                lines = text.strip().split("\n")
                if lines:
                    extracted["headers"] = [h.strip() for h in lines[0].split(",")]
                    extracted["row_count"] = max(0, len(lines) - 1)
            else:
                extracted["format"] = "text"
                extracted["line_count"] = len(text.split("\n"))

            return {
                "ocr_text": text,
                "extracted_data": extracted,
                "vision_description": "",
            }

        except Exception as e:
            return {"error": f"Text processing failed: {e}", "success": False}

    def _analyze_image_damage(self, image: Image.Image, ocr_text: str) -> str:
        """
        Analyze an image for damage assessment using basic image processing.
        For production, this would use a vision AI model.
        """
        width, height = image.size
        avg_color = self._get_average_color(image)
        brightness = self._get_brightness(image)

        description_parts = [f"Image dimensions: {width}x{height} pixels."]

        # Analyze based on image properties
        if brightness < 50:
            description_parts.append("The image appears very dark, possibly indicating low-light conditions or shadowed damage.")
        elif brightness < 100:
            description_parts.append("The image has moderate lighting conditions.")

        # Color analysis for damage detection
        r, g, b = avg_color
        if r > 200 and g < 100 and b < 100:
            description_parts.append("Detected significant red/heat signatures. Possible blood, fire damage, or brake light indicators.")
        if self._is_low_contrast(image):
            description_parts.append("Image has low contrast which may indicate smoke, fog, or overexposure at the scene.")

        if ocr_text:
            description_parts.append(f"OCR detected text: {ocr_text[:200]}...")

        description_parts.append("Damage assessment: Preliminary analysis based on available image data.")

        return " ".join(description_parts)

    def _get_average_color(self, image: Image.Image) -> Tuple[int, int, int]:
        """Get the average RGB color of an image."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        pixels = list(image.getdata())
        r_total = sum(p[0] for p in pixels)
        g_total = sum(p[1] for p in pixels)
        b_total = sum(p[2] for p in pixels)
        n = len(pixels)
        return (r_total // n, g_total // n, b_total // n)

    def _get_brightness(self, image: Image.Image) -> float:
        """Get the average brightness of an image (0-255)."""
        if image.mode != "L":
            image = image.convert("L")
        pixels = list(image.getdata())
        return sum(pixels) / len(pixels)

    def _is_low_contrast(self, image: Image.Image) -> bool:
        """Check if an image has low contrast."""
        if image.mode != "L":
            image = image.convert("L")
        pixels = list(image.getdata())
        std = (sum((p - sum(pixels) / len(pixels)) ** 2 for p in pixels) / len(pixels)) ** 0.5
        return std < 40

    def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """
        Extract structured data from OCR text using regex patterns.
        Looks for claim numbers, dates, amounts, names, etc.
        """
        extracted = {}

        # Find claim numbers
        claim_patterns = [
            r"CLM-\d{4}-\d{4}",
            r"Claim\s*(?:No|Number|#)[:\s]*([A-Z0-9-/]+)",
        ]
        for pattern in claim_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted["claim_numbers"] = matches
                break

        # Find policy numbers
        policy_matches = re.findall(r"POL-\d{4}-\d{4}", text)
        if policy_matches:
            extracted["policy_numbers"] = policy_matches

        # Find dates
        date_patterns = [
            r"\d{2}/\d{2}/\d{4}",
            r"\d{4}-\d{2}-\d{2}",
            r"\d{2}-\d{2}-\d{4}",
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text))
        if dates:
            extracted["dates_found"] = dates

        # Find monetary amounts
        amount_patterns = [
            r"EUR\s*[\d,]+\.?\d*",
            r"€\s*[\d,]+\.?\d*",
            r"[\d,]+\.\d{2}\s*(?:EUR|€)",
        ]
        amounts = []
        for pattern in amount_patterns:
            amounts.extend(re.findall(pattern, text))
        if amounts:
            extracted["amounts_found"] = amounts

        return extracted

    @staticmethod
    def generate_sample_claim_image(
        claim_number: str,
        policyholder: str,
        category: str,
        amount: float,
        output_path: str,
    ) -> str:
        """
        Generate a sample claim image with colored rectangles and text overlays.
        Used for demo/testing purposes.

        Returns the path to the generated image.
        """
        colors = {
            "auto": (70, 130, 180),      # Steel blue
            "health": (60, 179, 113),     # Medium sea green
            "property": (210, 105, 30),   # Chocolate
            "life": (147, 112, 219),      # Medium purple
            "travel": (255, 165, 0),      # Orange
            "accident": (220, 20, 60),    # Crimson
        }

        bg_color = colors.get(category, (100, 100, 100))

        img = Image.new("RGB", (800, 600), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # Header bar
        draw.rectangle([(0, 0), (800, 80)], fill=bg_color)
        draw.rectangle([(0, 80), (800, 82)], fill=(0, 0, 0, 50))

        # Damage visualization area
        draw.rectangle([(50, 120), (350, 420)], fill=(200, 200, 200), outline=(150, 150, 150))
        # Draw "damage" indicators
        for _ in range(5):
            import random
            x = random.randint(70, 330)
            y = random.randint(140, 400)
            size = random.randint(20, 60)
            draw.ellipse([(x, y), (x + size, y + size)], fill=(180, 80, 80), outline=(120, 40, 40))

        # Vehicle silhouette area
        draw.rectangle([(400, 120), (750, 420)], fill=(220, 220, 230), outline=(150, 150, 150))
        # Simple car shape
        draw.rectangle([(450, 250), (700, 350)], fill=bg_color, outline=(50, 50, 50))
        draw.rectangle([(470, 200), (680, 250)], fill=bg_color, outline=(50, 50, 50))
        draw.ellipse([(470, 340), (530, 380)], fill=(50, 50, 50))
        draw.ellipse([(620, 340), (680, 380)], fill=(50, 50, 50))
        # Damage indicator
        draw.ellipse([(550, 230), (620, 290)], fill=(180, 60, 60), outline=(100, 30, 30))

        # Text overlay
        try:
            font_large = ImageFont.truetype("arial.ttf", 28)
            font_medium = ImageFont.truetype("arial.ttf", 20)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except (IOError, OSError):
            font_large = ImageFont.load_default()
            font_medium = font_large
            font_small = font_large

        # Header text
        draw.text((30, 20), "BNP Paribas Cardif", fill="white", font=font_large)
        draw.text((30, 52), f"Claim: {claim_number}", fill="white", font=font_medium)

        # Info section
        draw.text((50, 440), f"Policyholder: {policyholder}", fill=(50, 50, 50), font=font_medium)
        draw.text((50, 470), f"Category: {category.upper()}", fill=(50, 50, 50), font=font_medium)
        draw.text((50, 500), f"Amount Claimed: EUR {amount:,.2f}", fill=(50, 50, 50), font=font_medium)
        draw.text((50, 530), f"Status: SUBMITTED", fill=(50, 50, 50), font=font_small)

        # Footer
        draw.rectangle([(0, 570), (800, 600)], fill=bg_color)
        draw.text((30, 576), "BNP Paribas Cardif - Claims Management System", fill="white", font=font_small)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        logger.info(f"Sample claim image generated: {output_path}")

        return output_path


# Module-level instance
processor = MultimodalProcessor()
