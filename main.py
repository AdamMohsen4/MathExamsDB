import argparse
import logging
from pdf_parser import extract_full_question_blocks_as_images


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract and organize math questions from a PDF exam.')
    parser.add_argument('pdf_path', help='Path to the input PDF file.') 
    parser.add_argument('--output_dir', default='output', help='Directory to save output files.')   
    #Increase resolution by setting the DPI
    parser.add_argument('--dpi', type=int, default=300, help='Resolution of the output images in DPI.')
    args = parser.parse_args()

    # Log the start of processing
    logger.info(f"Processing PDF: {args.pdf_path}")
    logger.info("Extracting question blocks as images...")
 
    # Extract question blocks as images
    extract_full_question_blocks_as_images(args.pdf_path, output_dir=f"{args.output_dir}/images", dpi=args.dpi)

if __name__ == '__main__':
    # Entry point of the script
    main()

