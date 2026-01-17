"""Image encoding and processing for multimodal pipeline."""
from typing import List, Optional, Union
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from config import settings


class ImageEncoder:
    """Handles image encoding using CLIP/SigLIP models."""
    
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """Initialize image encoder."""
        self.model = SentenceTransformer(model_name)
    
    def encode_image(self, image: Union[Image.Image, str, bytes]) -> np.ndarray:
        """Encode image to embedding vector."""
        if isinstance(image, str):
            # Base64 or file path
            if image.startswith("data:image"):
                image = self._decode_base64(image)
            else:
                image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        
        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        embedding = self.model.encode(image)
        return embedding
    
    def _decode_base64(self, base64_string: str) -> Image.Image:
        """Decode base64 image string."""
        header, encoded = base64_string.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(BytesIO(image_data))


class ImageCaptioner:
    """Generates captions for images using vision models."""
    
    def __init__(self):
        """Initialize captioner with vision model."""
        self.llm = ChatOpenAI(
            model=settings.vision_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
    
    def caption_image(self, image: Union[Image.Image, str, bytes], 
                     context: Optional[str] = None) -> str:
        """Generate caption for image."""
        # Convert image to base64 for API
        if isinstance(image, Image.Image):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
        elif isinstance(image, str):
            if not image.startswith("data:image"):
                with open(image, "rb") as f:
                    img_str = base64.b64encode(f.read()).decode()
            else:
                img_str = image.split(",")[1]
        else:
            img_str = base64.b64encode(image).decode()
        
        prompt = f"""Analyze this image and provide a detailed caption.
        {f'Context: {context}' if context else ''}
        Describe what you see, including objects, actions, text (OCR), and context."""
        
        # For GPT-4V, we'd use vision API here
        # Simplified for now - in production use proper vision API
        return "Image analysis placeholder - implement with GPT-4V API"


class MultimodalFusion:
    """Handles fusion of text and image embeddings."""
    
    def __init__(self, strategy: str = "early"):
        """
        Initialize fusion strategy.
        
        Args:
            strategy: 'early' (embedding fusion) or 'late' (textual grounding)
        """
        self.strategy = strategy
        self.image_encoder = ImageEncoder()
    
    def fuse(self, text: str, image: Optional[Union[Image.Image, str, bytes]] = None) -> np.ndarray:
        """
        Fuse text and image representations.
        
        Args:
            text: Text input
            image: Optional image input
            
        Returns:
            Fused embedding vector
        """
        if self.strategy == "early":
            return self._early_fusion(text, image)
        elif self.strategy == "late":
            return self._late_fusion(text, image)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")
    
    def _early_fusion(self, text: str, image: Optional[Union[Image.Image, str, bytes]]) -> np.ndarray:
        """Early fusion: combine embeddings."""
        from sentence_transformers import SentenceTransformer
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        text_emb = text_model.encode(text)
        
        if image is not None:
            img_emb = self.image_encoder.encode_image(image)
            # Concatenate and normalize
            fused = np.concatenate([text_emb, img_emb])
            return fused / np.linalg.norm(fused)
        
        return text_emb
    
    def _late_fusion(self, text: str, image: Optional[Union[Image.Image, str, bytes]]) -> np.ndarray:
        """Late fusion: textual grounding."""
        # In late fusion, we generate text description from image
        # then combine with original text
        captioner = ImageCaptioner()
        
        if image is not None:
            caption = captioner.caption_image(image, context=text)
            combined_text = f"{text}\n[Image description: {caption}]"
        else:
            combined_text = text
        
        from sentence_transformers import SentenceTransformer
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
        return text_model.encode(combined_text)

