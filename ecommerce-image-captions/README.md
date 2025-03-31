# E-commerce Image Captioner

This application generates detailed product captions for e-commerce images using the Gemma model via Ollama. It includes both an interactive web interface and a command-line tool for batch processing.

## Features

- Product name and detailed description generation
- Key attribute detection (color, style, pattern, material, etc.)
- User needs analysis based on the Alibaba research paper
- Cross-sell product suggestions
- Interactive web interface
- Batch processing capability

## Prerequisites

1. Install [Ollama](https://ollama.ai/) on your Windows system
2. Pull the Gemma model:
   ```bash
   ollama pull gemma3:4b
   ```
3. Python 3.8 or higher

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Interactive Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

This will open a web interface where you can:
1. Upload product images
2. View generated captions
3. See product attributes
4. Explore cross-sell suggestions

### Batch Processing

Use the command-line tool to process multiple images:
```bash
python batch_process.py input_directory output.json
```

Options:
- `input_directory`: Directory containing product images
- `output.json`: Path to save the results
- `--host`: Ollama API host URL (default: http://localhost:11434)

## Output Format

The generated captions include:
- Product name
- Detailed description
- Key attributes with confidence scores
- User needs analysis
- Cross-sell product suggestions

## Notes

- Ensure Ollama is running before using the application
- Supported image formats: JPG, JPEG, PNG
- The application requires an active internet connection for the Ollama API 