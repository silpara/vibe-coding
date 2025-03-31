import base64
import json
import requests
from typing import List, Dict
from pydantic import BaseModel

class ProductAttribute(BaseModel):
    name: str
    value: str
    confidence: float

class CrossSellProduct(BaseModel):
    category: str
    product_name: str
    search_query: str
    description: str
    attributes: List[ProductAttribute]

class ProductCaption(BaseModel):
    product_name: str
    description: str
    attributes: List[ProductAttribute]
    user_needs: List[str]
    cross_sell_products: List[CrossSellProduct]

    def to_dict(self) -> Dict:
        return json.loads(self.model_dump_json())

class ImageCaptioner:
    def __init__(self, ollama_host: str = "http://localhost:11434", model: str = "gemma3"):
        self.ollama_host = ollama_host
        self.model = model

    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = [model["name"] for model in response.json()["models"]]
                return sorted(models)
            return []
        except Exception:
            return []

    def set_model(self, model: str):
        """Set the model to use for captioning."""
        self.model = model

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_caption(self, image_path: str) -> ProductCaption:
        base64_image = self._encode_image(image_path)
        
        system_prompt = """You are an AI assistant tasked with analyzing a product image and providing comprehensive information for e-commerce purposes. Your output should include:
1. A detailed description of the product.
2. A list of its key attributes.
3. The user needs that this product satisfies, expressed as short phrases (up to 4 words) that describe specific shopping scenarios.
4. At least 5 cross-selling product suggestions from different product categories.

To accomplish this, follow these steps:
1. Identify the Product:
- Determine what the product is from the image.
2. Generate Description:
- Provide a thorough description of the product, including all visible features and any inferred details based on common knowledge.
3. List Attributes:
- Identify and list key attributes such as color, size, material, brand (if visible), etc.
4. Determine User Needs:
- Think about the specific shopping scenarios or contexts in which this product would be needed or useful.
- User needs should be expressed as short phrases (up to 4 words), similar to e-commerce concepts in the AliCoCo cognitive concept net.
- Examples: 'outdoor barbecue,' 'Christmas gifts for grandpa,' 'keep warm for kids.'
5. Suggest Cross-Selling Products:
- Suggest at least 5 complementary products from DIFFERENT categories than the main product
- Focus on products that enhance or complete the usage scenario
- For each cross-sell product:
  * Provide a complete, standalone description that makes sense without referencing the main product
  * Include specific attributes that match or complement the style of the main product
  * Ensure search terms are specific enough to find similar items
- Examples:
  * For a blue formal dress:
    - Pearl Necklace Set:
      Description: "Elegant freshwater pearl necklace and earring set with sterling silver clasps, perfect for formal occasions"
      Attributes: {Color: "White/Silver", Style: "Classic", Material: "Pearl/Sterling Silver"}
    - Leather Clutch Bag:
      Description: "Minimalist leather evening clutch with gold-tone hardware and detachable chain strap"
      Attributes: {Color: "Navy Blue", Material: "Genuine Leather", Style: "Modern"}
  * For a gaming laptop:
    - Gaming Headset:
      Description: "Professional-grade gaming headset with 7.1 surround sound and RGB lighting"
      Attributes: {Color: "Black/RGB", Style: "Gaming", Features: "Noise Cancelling"}
    - Ergonomic Chair:
      Description: "High-back gaming chair with lumbar support and adjustable armrests"
      Attributes: {Style: "Gaming/Ergonomic", Material: "Mesh/PU Leather"}

Remember to maintain style consistency across suggested products while ensuring each description stands independently.

Important Guidelines for Confidence Scores:
- Use 0.8-0.9 only for attributes directly visible in the image (e.g., main color, category)
- Use 0.5-0.7 for attributes that can be reasonably inferred (e.g., material type based on texture)
- Use 0.3-0.4 for attributes that are partially visible or uncertain
- Use 0.0 and "Not visible" for attributes that cannot be determined from the image

Respond with a JSON object in the following format:
{
    "product_name": "Specific product name with key visible characteristics",
    "description": "Detailed description of visible features and characteristics",
    "attributes": [
        {
            "name": "Color",
            "value": "Specific color or 'Not visible'",
            "confidence": 0.0-0.9
        },
        {
            "name": "Pattern",
            "value": "Pattern description or 'Not visible'",
            "confidence": 0.0-0.9
        },
        {
            "name": "Material",
            "value": "Material type or 'Not visible'",
            "confidence": 0.0-0.9
        },
        {
            "name": "Style",
            "value": "Style description or 'Not visible'",
            "confidence": 0.0-0.9
        },
        {
            "name": "Gender",
            "value": "Target gender or 'Unisex'",
            "confidence": 0.0-0.9
        },
        {
            "name": "Category",
            "value": "Product category",
            "confidence": 0.0-0.9
        }
    ],
    "user_needs": [
        "specific shopping scenario",
        "usage context",
        "occasion description"
    ],
    "cross_sell_products": [
        {
            "category": "Different category than main product",
            "product_name": "Specific complementary product",
            "search_query": "Specific search terms to find this product",
            "description": "Standalone description of the product's features and benefits",
            "attributes": [
                {
                    "name": "Color",
                    "value": "Color that complements main product",
                    "confidence": 0.8
                },
                {
                    "name": "Style",
                    "value": "Style that matches main product theme",
                    "confidence": 0.7
                },
                {
                    "name": "Material",
                    "value": "Material appropriate for the product",
                    "confidence": 0.7
                },
                {
                    "name": "Features",
                    "value": "Key product features",
                    "confidence": 0.8
                },
                {
                    "name": "Occasion",
                    "value": "Suitable usage occasions",
                    "confidence": 0.7
                }
            ]
        }
    ]
}

Note: For each cross-sell product:
1. Provide a complete set of relevant attributes that make sense for that product category
2. Ensure descriptions are standalone and comprehensive
3. Include style-matching attributes that complement the main product
4. Maintain high specificity in product names and search queries
5. Provide at least 5 diverse cross-sell suggestions from different categories"""

        user_prompt = "Analyse the input product image and respond using JSON"

        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json={
                "model": self.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "images": [base64_image],
                "stream": False,
                "temperature": 0.1
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from Ollama API: {response.text}")

        # Parse the response and convert to ProductCaption
        try:
            response_text = response.json()["response"]
            # Extract JSON content between curly braces
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                caption_data = json.loads(json_content)
                return ProductCaption(**caption_data)
            else:
                raise Exception("No valid JSON found in response")
        except Exception as e:
            raise Exception(f"Error parsing model response: {str(e)}\nResponse: {response_text}")

    def process_directory(self, input_dir: str, output_file: str):
        """Process all images in a directory and save results to a JSON file."""
        import os
        import glob

        results = []
        image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                     glob.glob(os.path.join(input_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(input_dir, "*.png"))

        for image_file in image_files:
            try:
                caption = self.generate_caption(image_file)
                results.append({
                    "image_path": image_file,
                    "caption": caption.to_dict()
                })
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2) 