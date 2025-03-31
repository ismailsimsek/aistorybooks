import shutil
import statistics
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import streamlit as st
from llama_index.core.schema import Document
from streamlit.runtime.uploaded_file_manager import UploadedFile

from aistorybooks.phidataa.classic_stories import PhiStoryBookGenerator


@dataclass
class AppInputs:
    """
    Data class to hold the input values for the Streamlit app.
    """

    uploaded_file: UploadedFile | None = None
    language: str = "German"
    level: str = "B1 Intermediate"
    summary_size: str = "Long (150 sentences/1200 words)"
    writing_style: str = "Philosophical"
    chunk_size: int = 10
    padding: int = 1
    skip_first_n_pages: int = 0
    language_options: List[str] = field(
        default_factory=lambda: ["German", "English", "Spanish", "French"]
    )
    level_options: List[str] = field(
        default_factory=lambda: [
            "A1 Beginner",
            "A2 Elementary",
            "B1 Intermediate",
            "B2 Upper Intermediate",
            "C1 Advanced",
            "C2 Proficiency",
        ]
    )
    summary_size_options: List[str] = field(
        default_factory=lambda: [
            "Short (50 sentences/400 words)",
            "Medium (100 sentences/800 words)",
            "Long (150 sentences/1200 words)",
        ]
    )
    writing_style_options: List[str] = field(
        default_factory=lambda: [
            "Philosophical",
            "Narrative",
            "Descriptive",
            "Humorous",
            "Formal",
        ]
    )


def st_sidebar(inputs: AppInputs):
    """
    Creates the sidebar for the Streamlit app and populates the input values.

    Args:
        inputs (AppInputs): An instance of the AppInputs data class.
    """

    st.header("Input Options:")
    with st.form(key='inputs_form', border=False):
        inputs.uploaded_file = st.file_uploader(
            "Upload your novel (PDF)",
            type=["pdf"],
            accept_multiple_files=False,
            help="Upload the PDF file of the novel you want to convert.",
        )

        inputs.language = st.selectbox(
            "Select Target Story Language",
            inputs.language_options,
            index=inputs.language_options.index(inputs.language),
            help="Choose the language you want the storybook to be in.",
        )

        inputs.level = st.selectbox(
            "Select Language Level",
            inputs.level_options,
            index=inputs.level_options.index(inputs.level),
            help="Select the target language proficiency level for the storybook.",
        )

        inputs.summary_size = st.selectbox(
            "Desired Summary Length (Per Chunk)",
            inputs.summary_size_options,
            index=inputs.summary_size_options.index(inputs.summary_size),
            help="Specify the desired length of the summary for each chunk of the novel.",
        )

        inputs.writing_style = st.selectbox(
            "Desired Writing Style",
            inputs.writing_style_options,
            index=inputs.writing_style_options.index(inputs.writing_style),
            help="Choose the writing style for the generated storybook.",
        )

        inputs.chunk_size = st.number_input(
            "Chunk Size",
            min_value=1,
            value=inputs.chunk_size,
            help="Number of pages to process per iteration. Larger chunks may take longer to process.",
        )
        inputs.padding = st.number_input(
            "Padding",
            min_value=0,
            value=inputs.padding,
            help="Number of pages to overlap between chunks. Helps maintain context.",
        )
        inputs.skip_first_n_pages = st.number_input(
            "Skip First N Pages",
            min_value=0,
            value=inputs.skip_first_n_pages,
            help="Number of pages to skip at the beginning of the novel (e.g., table of contents).",
        )
        submit_button = st.form_submit_button(label='Submit')
        return submit_button


def st_process_file(inputs: AppInputs) -> List[List[Document]]:
    uploaded_file_name = inputs.uploaded_file.name
    st.info(f"Uploaded File: **{inputs.uploaded_file.name}**. Preparing your storybook... (Working in the background)."
            f"  \nPlease note: Processing is powered by the free tier of Gemini, which may experience rate limiting.",
            icon=":material/info:")

    try:
        temp_folder = Path(tempfile.mkdtemp(prefix="story_gen_temp_"))
        progress_value = 0
        progress = st.progress(value=progress_value, text=f"Processing file...")
        pdf_file = temp_folder.joinpath(uploaded_file_name)
        md_file_name = f"{pdf_file.stem}.md"
        # pdf_file_final = pdf_file.parent.joinpath(f"{pdf_file.stem}_story.pdf")
        pdf_file.write_bytes(inputs.uploaded_file.getvalue())
        generator = PhiStoryBookGenerator(
            language=inputs.language,
            level=inputs.level,
            summary_size=inputs.summary_size,
            writing_style=inputs.writing_style,
        )
        st.session_state[md_file_name] = ""
        button_container = st.empty()
        info_container = st.empty()
        it = generator.run(pdf_file=pdf_file,
                           chunk_size=inputs.chunk_size,
                           padding=inputs.padding,
                           skip_first_n_pages=inputs.skip_first_n_pages
                           )
        for response in it:
            if response.event == "RunFailed":
                st.error(f"{response.content}", icon=":material/error:")
                progress.progress(value=progress_value, text=f"Error: {response.content}")
            else:
                progress_value = response.metrics['progress_percent']
                progress.progress(value=progress_value, text=response.metrics['progress_info'])
            st.session_state[md_file_name] += f"\n\n{response.content}"
            button_container.empty()
            button_container.download_button(label='Download Storybook as Markdown',
                                             data=st.session_state.get(md_file_name),
                                             file_name=md_file_name,
                                             mime='text/markdown',
                                             on_click="ignore",
                                             key=str(uuid.uuid4()),
                                             type="primary",
                                             icon=":material/download:",
                                             )
            info_container.empty()
            metrics = generator.model.metrics
            avg_response_time = statistics.mean(metrics.get('response_times', [])) if metrics else 0
            input_tokens = metrics.get('input_tokens', 0) if metrics else 0
            output_tokens = metrics.get('output_tokens', 0) if metrics else 0
            total_tokens = metrics.get('total_tokens', 0) if metrics else 0
            info_container.info(f"Model: {generator.model.name} "
                                f"  \n Avg response time: {avg_response_time} "
                                f"  \n Input tokens: {input_tokens} "
                                f"  \n Output tokens: {output_tokens} "
                                f"  \n Total tokens: {total_tokens}",
                                icon=":material/info:")
    finally:
        if temp_folder and temp_folder.exists():
            shutil.rmtree(temp_folder)
            st.info(f"Temporary folder and its contents cleared", icon=":material/info:")


def st_main_page(inputs: AppInputs):
    """
    Creates the main page for the Streamlit app and displays the input values.

    Args:
        inputs (AppInputs): An instance of the AppInputs data class.
    """
    st.title("Novel to Storybook Generator")
    st.write("---")
    options_text = f"""
        **Selected Options:**  
        **Language:** {inputs.language} | 
        **Language Level:** {inputs.level} | 
        **Summary Length:** {inputs.summary_size} | 
        **Writing Style:** {inputs.writing_style} | 
        **Chunk Size:** {inputs.chunk_size} | 
        **Padding:** {inputs.padding} | 
        **Skip First N Pages:** {inputs.skip_first_n_pages}
        """
    st.markdown(options_text)
    if inputs.uploaded_file:
        st_process_file(inputs=inputs)


def st_set_css_and_footer():
    st.markdown(
        """
            <style>
                .stAppHeader {
                    background-color: rgba(255, 255, 255, 0.0);  /* Transparent background */
                    visibility: visible;  /* Ensure the header is visible */
                }
                
                .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
                .footer {
                    position: fixed;
                    left: 0;
                    bottom: 2px;
                    width: 100%;
                    text-align: right;
                    padding-right: 40px;
                }
                .footer a {
                    margin: 0 5px; /* Reduced margin for smaller icons */
                    text-decoration: none;
                }

                .footer img {
                    height: 18px; /* Adjusted height for smaller icons */
                    width: 18px; /* Adjusted width for smaller icons */
                    vertical-align: middle;
                    opacity: 0.7; /* Added opacity for a softer look */
                    transition: opacity 0.3s ease; /* Added transition for hover effect */
                }

                .footer img:hover {
                    opacity: 1.0; /* Increased opacity on hover */
                }
            </style>
            """,
        unsafe_allow_html=True,
    )

    footer_html = """
    <div class='footer'>
        <p>
            Need to leverage AI for your business, improve your data and analytics setup or strategy? 
            Feel free to contact. 
        <a href="https://github.com/ismailsimsek" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/ismailsimsek/" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
        </a>
        <a href="https://medium.com/@ismail-simsek" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/1384/1384015.png" alt="Medium">
        </a>
        Copyright 2025 Ismail Simsek</p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def main():
    st.set_page_config(layout="wide")
    st_set_css_and_footer()
    inputs = AppInputs()
    with st.sidebar:
        st_sidebar(inputs)

    st_main_page(inputs)


if __name__ == "__main__":
    main()
