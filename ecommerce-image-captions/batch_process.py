import argparse
import os
from pathlib import Path
from image_captioner import ImageCaptioner

def main():
    parser = argparse.ArgumentParser(description="Batch process images for e-commerce product captioning")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API host URL")
    parser.add_argument("--model", default="gemma3", help="Model to use for captioning")
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    input_dir = os.path.abspath(args.input_dir)
    output_file = os.path.abspath(args.output_file)
    
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize captioner
    captioner = ImageCaptioner(ollama_host=args.host, model=args.model)
    
    print(f"Using model: {args.model}")
    print(f"Processing images from: {input_dir}")
    print(f"Results will be saved to: {output_file}")
    
    # Get list of unique image files using a set
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = set()
    for file in Path(input_dir).rglob('*'):
        if file.suffix.lower() in extensions:
            image_files.add(file)
    
    # Convert to sorted list for consistent ordering
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"\nFound {len(image_files)} images to process")
    
    # Process images
    results = []
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"\nProcessing image {i}/{len(image_files)}: {image_path}")
            caption = captioner.generate_caption(str(image_path))
            results.append({
                "image_path": str(image_path),
                "caption": caption.to_dict()
            })
            print(f"✓ Successfully processed {image_path}")
        except Exception as e:
            print(f"✗ Error processing {image_path}: {str(e)}")
    
    # Save results
    if results:
        try:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nSuccessfully processed {len(results)} images")
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"\nError saving results to {output_file}: {str(e)}")
    else:
        print("\nNo images were successfully processed")

if __name__ == "__main__":
    main() 