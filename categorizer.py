from PIL import Image
import pytesseract
import os

def extract_text_from_image(image_path, output_dir):
    """
    Extracts text from an image using Tesseract OCR with the math language model and saves it to a file.

    Args:
        image_path (str): Path to the input image file.
        output_dir (str): Directory to save the extracted text file.

    Returns:
        str: Extracted text from the image.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Open the image file
        image = Image.open(image_path)

        # Use Tesseract to extract text from the image with the math language model
        text = pytesseract.image_to_string(image, lang='swe+math')

        # Check if any text was extracted
        if not text:
            print("No text was extracted from the image.")
            return ""

        # Save the extracted text to a file
        base_name = os.path.basename(image_path)
        text_file_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)

        return text

    except Exception as e:
        print(f"An error occurred while extracting text from the image: {e}")
        return ""