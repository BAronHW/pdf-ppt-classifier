import os
import numpy as np
import pymupdf
from PIL import Image
from typing import Tuple

# """" This module extracts the features of a single page and the idea is to be able to extract the average RGB values of all documents and pdf's as well as their word-counts to act as features for the machine learning model to learn from """"

def extractRGBFeatures(document_path) -> Tuple[float, float, float]:

    pdf = pymupdf.open(document_path) 
    pdfpagenum = (pdf.page_count - 1)
    total_rgb = np.array([0.0, 0.0, 0.0])
    for page_num in range(pdfpagenum):
        page = pdf.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_array = np.array(img)

        avg_rgb = img_array.mean(axis=(0,1))
        total_rgb += avg_rgb
    
    avg_rgb = total_rgb / pdfpagenum

    return avg_rgb

def extractWordCount(document_path) -> int:

    pdf = pymupdf.open(document_path)
    pdfpagenum = (pdf.page_count - 1)
    totalwords = 0
    for page in pdf:
        text = page.get_text().encode("utf-8")
        textlen = len(text)
        totalwords += textlen
    
    if totalwords == 0:
        return 0
    meanwords = totalwords/pdfpagenum
        
    return meanwords


examplepdf = r"C:\Users\Aaron\OneDrive - Lancaster University\Documents\my projects\pdf-ppt-classifier\data\documents\0wlz9DKaxwQIPqkAU1BicYG7XHbPxy.pdf"
exampleppt = r"C:\Users\Aaron\OneDrive - Lancaster University\Documents\my projects\pdf-ppt-classifier\data\powerpoints\1VYDQOEhrEkqg4c3IayMEDaEvg46Lc.pdf"

print(extractWordCount(examplepdf))
print(extractRGBFeatures(examplepdf))