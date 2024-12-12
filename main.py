import argparse
import logging
import os
from pdf_parser import extract_full_question_blocks_as_images
from categorizer import extract_text_from_image

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

    # Log the start of processing
    logger.info(f"Processing PDF: {args.pdf_path}")
    logger.info("Extracting question blocks as images...")

    # Extract question blocks as images
    extract_full_question_blocks_as_images(args.pdf_path, output_dir=f"{args.output_dir}/images", dpi=args.dpi)

    # Extract text from images
    logger.info("Extracting text from images...")
    image_output_dir = f"{args.output_dir}/images"
    text_output_dir = f"{args.output_dir}/text"
    for image_file in os.listdir(image_output_dir):
        image_path = os.path.join(image_output_dir, image_file)
        extract_text_from_image(image_path, output_dir=text_output_dir)

if __name__ == '__main__':
    # Entry point of the script
    main()