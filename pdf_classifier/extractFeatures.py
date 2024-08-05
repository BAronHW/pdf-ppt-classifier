import numpy as np
from PIL import Image
import pymupdf
from typing import Tuple

def extractRGBFeatures(document_path: str) -> Tuple[float, float, float]:
    try:
        pdf = pymupdf.open(document_path)
        pdfpagenum = max(1, pdf.page_count)
        total_rgb = np.array([0.0, 0.0, 0.0])
        valid_pages = 0

        for page_num in range(pdfpagenum):
            try:
                page = pdf.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)
                if img_array.size > 0:
                    avg_rgb = img_array.mean(axis=(0,1))
                    if not np.isnan(avg_rgb).any() and not np.isinf(avg_rgb).any():
                        total_rgb += avg_rgb
                        valid_pages += 1
            except Exception as e:
                print(f"Error processing page {page_num} in {document_path}")
                continue

        if valid_pages > 0:
            avg_rgb = (total_rgb / valid_pages) / 255
            return tuple(np.clip(avg_rgb, 0, 1))  # Ensure values are between 0 and 1
        else:
            print(f"No valid pages found in {document_path}")
            return (np.nan, np.nan, np.nan)

    except Exception as e:
        print(f"Error in extractRGBFeatures for {document_path}: {str(e)}")
        return (np.nan, np.nan, np.nan)

# def extractWordCount(document_path: str) -> float:
#     try:
#         pdf = pymupdf.open(document_path)
#         pdfpagenum = max(1, pdf.page_count)
#         totalwords = 0
#         valid_pages = 0

#         for page in pdf:
#             try:
#                 text = page.get_text()
#                 words = text.split()
#                 totalwords += len(words)
#                 valid_pages += 1
#             except Exception as e:
#                 print(f"Error processing page for word count in {document_path}: {str(e)}")
#                 continue

#         if valid_pages > 0:
#             meanwords = totalwords / valid_pages
#             return float(meanwords)
#         else:
#             print(f"No valid pages found for word count in {document_path}")
#             return np.nan

#     except Exception as e:
#         print(f"Error in extractWordCount for {document_path}: {str(e)}")
#         return np.nan
