import markdown
import os
import pdfkit
import re
import requests
from openai import OpenAI
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import Type

from config import Config
from crewai.tools import BaseTool


class ImageGeneratorInput(BaseModel):
    """Input schema for ImageGenerator."""

    image_description: str = Field(
        ..., description="A detailed description of the image to be generated."
    )


class ImageGenerator(BaseTool):
    """
    Generates an image for a given description using the OpenAI API.
    """

    name: str = "Generate Image"
    description: str = (
        "Generates an image for a given description using the OpenAI image generation API, "
        "saves it in the current folder, and returns the image path. "
        "The image description should be detailed and longer than 100 characters."
    )
    args_schema: Type[BaseModel] = ImageGeneratorInput
    client: OpenAI = Field(
        default_factory=lambda: OpenAI(api_key=Config.OPENAI_API_KEY),
        description="OpenAI client instance.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, image_description: str) -> str:
        if len(image_description) < 100:
            raise ValueError("Please provide a longer image description (at least 100 characters).")

        response = self.client.images.generate(
            model="dall-e-3",
            prompt=f"Image is about: {image_description}. "
                   f"Style: Illustration. Create an illustration incorporating a vivid palette with an emphasis on shades "
                   f"of azure and emerald, augmented by splashes of gold for contrast and visual interest. The style "
                   f"should evoke the intricate detail and whimsy of early 20th-century storybook illustrations, "
                   f"blending realism with fantastical elements to create a sense of wonder and enchantment. The "
                   f"composition should be rich in texture, with a soft, luminous lighting that enhances the magical "
                   f"atmosphere. Attention to the interplay of light and shadow will add depth and dimensionality, "
                   f"inviting the viewer to delve into the scene. DON'T include ANY text in this image. DON'T include "
                   f"colour palettes in this image.",
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        words = image_description.split()[:5]
        safe_words = [re.sub(r"[^a-zA-Z0-9_]", "", word) for word in words]
        filename = "_".join(safe_words).lower() + ".png"
        filepath = Path(os.getcwd()).joinpath("images", filename)

        # Download the image from the URL
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            print(f"Saving image: {filepath.as_posix()}")
            filepath.write_bytes(image_response.content)
        else:
            print("Failed to download the image.")
            return ""

        return filepath.relative_to(os.getcwd()).as_posix()


class MarkdownToPdfConverterInput(BaseModel):
    """Input schema for MarkdownToPdfConverter."""

    markdown_file_name: str = Field(
        ..., description="Path to the input Markdown file."
    )


class MarkdownToPdfConverter(BaseTool):
    """
    Converts a Markdown file to a PDF document using the pdfkit python library.
    """

    name: str = "Convert Markdown to PDF"
    description: str = (
        "Converts a Markdown file to a PDF document using the pdfkit python library. "
        "The input should be a valid path to a markdown file."
    )
    args_schema: Type[BaseModel] = MarkdownToPdfConverterInput
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, markdown_file_name: str) -> str:
        output_file = os.path.splitext(markdown_file_name)[0] + ".pdf"

        # Read the markdown file
        with open(Path(markdown_file_name), "r") as f:
            text = f.read()
        # Convert to HTML
        html = markdown.markdown(text)
        # Convert to PDF
        pdfkit.from_string(html, output_file, options={"enable-local-file-access": ""})

        return output_file
