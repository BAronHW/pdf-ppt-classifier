import numpy as np
from PIL import Image
import pymupdf
import cv2
from typing import Tuple

def extractRGBFeatures(document_path: str) -> Tuple[float, float, float]:
    try:
        print("now extracting features")
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
            except pymupdf.FileDataError as e:
                print(f"Error processing page {page_num} in {document_path}")
                continue

        if valid_pages > 0:
            avg_rgb = (total_rgb / valid_pages) / 255
            return tuple(np.clip(avg_rgb, 0, 1)) #make sure between 0 and 1
        else:
            print(f"No valid pages found in {document_path}")
            return (np.nan, np.nan, np.nan)

    except Exception as e:
        print(f"Error in extractRGBFeatures for {document_path}: {str(e)}")
        return (np.nan, np.nan, np.nan)
    

def extractEdges(document_path: str) -> Tuple[float, float]:
    # takes the document path then opens the document using pymupdf
    # loop through the document
    # for every page turn the page into a pixmap then from a pixmap to an img 
    # apply a gaussian filter to the image to remove noise
    # apply canny edge detector and then find the ratio of edges to non edges and the average edge intencity
    try:
        pdf = pymupdf.open(document_path) 
        pdfpagenum = max(1, pdf.page_count)
        total_edge_ratio = 0.0
        total_edge_intensity = 0.0
        valid_pages = 0

        for page_num in range(pdfpagenum):
            try:
                page = pdf.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img_array = np.array(img)
        
                if img_array.size > 0:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blurred, 100, 200)
                    edge_ratio = np.sum(edges > 0) / edges.size
                    edge_intensity = np.mean(edges)

                    total_edge_ratio += edge_ratio
                    total_edge_intensity += edge_intensity
                    valid_pages += 1

            except Exception as e:
                print(e)
                continue

        if valid_pages > 0:
            avg_edge_ratio = total_edge_ratio / valid_pages
            avg_edge_intensity = total_edge_intensity / valid_pages
            return (avg_edge_ratio, avg_edge_intensity)
        else:
            print(f"No valid pages found in {document_path}")
            return (np.nan, np.nan)
        
    except Exception as e:
        print(e)
        return (np.nan,np.nan)