import os
import re
from pathlib import Path

import markdown
import pdfkit
import requests
from crewai_tools import tool
from openai import OpenAI

from config import Config


@tool
def generate_image(image_description: str) -> str:
    """
    Generates an image for a given description.
    Using the OpenAI image generation API,
    saves it in the current folder, and returns the image path.
    """
    if len(image_description) < 100:
        raise Exception("Please provide longer image description")

    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    response = client.images.generate(
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
    # print(f"Image content: {content}")
    image_url = response.data[0].url
    words = image_description.split()[:5]
    safe_words = [re.sub(r'[^a-zA-Z0-9_]', '', word) for word in words]
    filename = "_".join(safe_words).lower() + ".png"
    filepath = Path(os.getcwd()).joinpath('images', filename)

    # Download the image from the URL
    image_response = requests.get(image_url)
    if image_response.status_code == 200:
        print(f"Saving image: {filepath.as_posix()}")
        filepath.write_bytes(image_response.content)
    else:
        print("Failed to download the image.")
        return ""

    return filepath.relative_to(os.getcwd()).as_posix()
    # return filepath.as_posix()


@tool
def convert_markdown_to_pdf(markdown_file_name: str) -> str:
    """
    Converts a Markdown file to a PDF document using the pdfkit python library.

    Args:
        markdown_file_name (str): Path to the input Markdown file.

    Returns:
        str: Path to the generated PDF file.
    """
    output_file = os.path.splitext(markdown_file_name)[0] + '.pdf'

    # Read the markdown file
    with open(Path(markdown_file_name), 'r') as f:
        text = f.read()
    # Convert to HTML
    html = markdown.markdown(text)
    # Convert to PDF
    pdfkit.from_string(html, output_file, options={"enable-local-file-access": ""})

    return output_file
