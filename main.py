import argparse
import logging
import os
from image_extractor import extract_full_question_blocks_as_images
from text_extractor import extract_text_from_image
from categorizer import load_model, categorize_questions, initialize_database, link_questions_to_images

def main():
    """
    Main function to extract and organize math questions from a PDF exam.
    It extracts question blocks as images and then extracts text from those images.
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract and organize math questions from a PDF exam.')
    parser.add_argument('pdf_path', help='Path to the input PDF file.')
    parser.add_argument('--output_dir', default='output', help='Directory to save output files.')
    parser.add_argument('--dpi', type=int, default=300, help='Resolution of the output images in DPI.')
    args = parser.parse_args()

    logger.info(f"Processing PDF: {args.pdf_path}")
    logger.info("Extracting question blocks as images...")
    question_images = extract_full_question_blocks_as_images(args.pdf_path, output_dir=f"{args.output_dir}/images", dpi=args.dpi)

    # Extract text from images
    logger.info("Extracting text from images...")
    image_output_dir = f"{args.output_dir}/images"
    text_output_dir = f"{args.output_dir}/text"
    extracted_texts = []
    for image_file in os.listdir(image_output_dir):
        image_path = os.path.join(image_output_dir, image_file)
        text = extract_text_from_image(image_path, output_dir=text_output_dir)
        extracted_texts.append(text)

    # Load the model and tokenizer
    logger.info("Loading model and tokenizer...")
    tokenizer, model = load_model()

    # Define the label map
    label_map = {
        0: "Least squares method",
        1: "Distance calculation",
        2: "Linear transformation",
        3: "Matrix multiplication",
        4: "Determinant calculation",
        5: "Inverse calculation",
        6: "Eigenvalue calculation",
        7: "Reflection"
    }

    # Categorize the extracted text
    logger.info("Categorizing questions...")
    categorized = {}
    for text in extracted_texts:
        categories = categorize_questions(text, tokenizer, model, label_map)
        categorized.update(categories)

    # Link questions to images
    logger.info("Linking questions to images...")
    initialize_database()
    link_questions_to_images(categorized)

if __name__ == '__main__':
    main()