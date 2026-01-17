"""Vision agent for image analysis and understanding."""
from typing import Dict, Any, Optional, Union
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from multimodal import ImageEncoder, ImageCaptioner, MultimodalFusion
from config import settings


class VisionAgent:
    """Handles image analysis and visual understanding."""
    
    def __init__(self):
        """Initialize vision agent."""
        self.llm = ChatOpenAI(
            model=settings.vision_model,
            temperature=settings.temperature,
            api_key=settings.openai_api_key
        )
        self.image_encoder = ImageEncoder()
        self.captioner = ImageCaptioner()
        self.fusion = MultimodalFusion(strategy="late")
    
    def analyze(self, query: str, image: Union[Image.Image, str, bytes], 
                fusion_strategy: str = "late") -> Dict[str, Any]:
        """
        Analyze image with text query.
        
        Args:
            query: Text query about the image
            image: Image to analyze
            fusion_strategy: 'early' or 'late' fusion
            
        Returns:
            Dict with 'response', 'caption', 'ocr_text'
        """
        # Generate caption
        caption = self.captioner.caption_image(image, context=query)
        
        # For OCR, we'd use a proper OCR library like Tesseract
        # Simplified for now
        ocr_text = "OCR placeholder - implement with Tesseract or similar"
        
        # Use fusion for multimodal understanding
        if fusion_strategy == "late":
            # Late fusion: combine text and image description
            combined_query = f"{query}\n[Image: {caption}]"
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a vision expert. Analyze images and answer questions about them.
                You have access to image captions and OCR text."""),
                ("user", "Query: {query}\n\nImage Caption: {caption}\nOCR Text: {ocr}\n\nAnswer:")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "query": query,
                "caption": caption,
                "ocr": ocr_text
            })
        else:
            # Early fusion: use embeddings
            embedding = self.fusion.fuse(query, image)
            # For early fusion, we'd typically use a multimodal model
            # Simplified here
            response = self.llm.invoke(f"Based on the image and query '{query}', provide analysis.")
        
        return {
            "response": response.content if hasattr(response, 'content') else str(response),
            "caption": caption,
            "ocr_text": ocr_text,
            "agent": "vision"
        }

