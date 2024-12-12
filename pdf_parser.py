import fitz  # PyMuPDF
import os
import re

def extract_full_question_blocks_as_images(pdf_path, output_dir='output/images', dpi=300):
    """
    Extracts question blocks from a PDF file and saves them as images.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save output images. Defaults to 'output/images'.
        dpi (int): Resolution of the output images in DPI. Defaults to 300.

    Returns:
        list: A list of file paths to the saved question images.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    question_images = []

    # Open the PDF document
    pdf_document = fitz.open(pdf_path)
    # Regular expression to match question numbers (e.g., "1. ", "2. ", etc.)
    question_pattern = re.compile(r'^\d+\.\s', re.MULTILINE)

    # Iterate through each page in the PDF
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        # Extract and sort text blocks by vertical position
        blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])

        question_bounds, current_bounds = [], None
        # Iterate through each text block
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]
            if question_pattern.match(text.strip()):  # Detect exercise delimiter (e.g., "1.")
                if current_bounds:
                    question_bounds.append(current_bounds)
                current_bounds = [x0, y0, x1, y1]
            elif current_bounds:
                current_bounds = [min(current_bounds[0], x0), min(current_bounds[1], y0), max(current_bounds[2], x1), y1]

        if current_bounds:
            question_bounds.append(current_bounds)

        # Extract images for each question block
        for idx, bounds in enumerate(question_bounds):
            # Increase resolution by setting the DPI
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix, clip=(bounds[0], bounds[1], bounds[2], bounds[3]))
            img_path = os.path.join(output_dir, f"question_page{page_number + 1}_q{idx + 1}.png")
            pix.save(img_path)
            question_images.append(img_path)

    pdf_document.close()
    return question_images