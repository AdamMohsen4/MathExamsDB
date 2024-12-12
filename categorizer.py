from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import os

def preprocess_image(image):
    """
    Preprocess the image to enhance OCR accuracy.
    
    Args:
        image (PIL.Image): The input image.
    
    Returns:
        PIL.Image: The preprocessed image.
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Apply thresholding
    image = image.point(lambda x: 0 if x < 140 else 255, '1')
    
    # Resize the image
    base_width = 1800
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    image = image.resize((base_width, h_size), Image.LANCZOS)
    
    return image

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

        # Preprocess the image
        image = preprocess_image(image)

        # Use Tesseract to extract text from the image with the math language model
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, lang='swe+math', config=custom_config)

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