# AI-Powered Storybook and Poem Generation for Language Learners

This project explores the use of AI agents and various agentic frameworks to create engaging learning materials. It
experiments with different agentic flows and tool integrations to enhance the learning experience.

The primary focus is to assist language learners by generating storybooks and poems tailored to their specific needs.
For example, users can transform classic novels into storybooks adapted for language learning, customized to their
desired proficiency level.

You can interact with and use the live application directly on [Hugging Face Spaces](https://huggingface.co/spaces/ismailsimsek/aistorybooks)

## Classics to Story Book Generators

This tools simplifies classic novels into engaging storybooks. You can choose the target language, reading level,
and even writing style and saves the final product as a convenient markdown file.

Perfect for language learners to experience the joy of classic stories in a language that supports their learning
journey.

### Classics to Story Book Generator V2

This version of the storybook generator takes a PDF file of a classic novel as input and transforms it into a storybook
tailored to your specifications. It leverages advanced language processing to summarize the content, adjust the reading
level, and adapt the writing style to your preferences. This version is designed to work directly with the content of
the provided PDF.

**Key Features:**

* **PDF Input:** Accepts a PDF file of a classic novel as the primary input source.
* **Customizable Output:** Allows you to specify the target language, reading level (e.g., A1 Intermediate), summary
  size (e.g., Long with 150 sentences/1200 words), and writing style (e.g., Philosophical).
* **Chunk-Based Processing:** Processes the PDF content in chunks, allowing for efficient handling

```python
from aistorybooks.phidataa.classic_stories import PhiStoryBookGenerator
from pathlib import Path

generator = PhiStoryBookGenerator(
    language="German",
    level="A1 Intermediate",
    summary_size="Long (150 sentences/1200 words)",
    writing_style="Philosophical",
)
pdf_file = Path("The-Brothers-Karamazov.pdf")
results = generator.run(pdf_file=pdf_file, chunk_size=1, padding=0, skip_first_n_pages=0)

for response in results:
    print(response.content)

```

### Classics to Story Book Generator V1

This version uses llm knowledge to generate the story, and it adds llm generated illustration related to story content,
to the final markdown file.

Example Output:

- [story.md](story.md)

Usage:

```python
from aistorybooks.crewaia.classic_stories import StoryBookGenerator

generator = StoryBookGenerator(book="The Karamazov Brothers",
                               author="Fyodor Dostoevsky",
                               language="German",
                               level="A2 Beginner",
                               summary_size="10 Chapters, each chapter more than 100 sentences log",
                               writing_style="Philosophical")
generator.generate()
```

## Classics to Poem Generator

Transforms a classic book into a poem and saves it as markdown and PDF, with and llm generated illustration.

**Specify the poem style**: Mention the specific poetic style (e.g., "Alexander Pushkin") to get specific poetic style.

Example Output:

- [poem.md](poem.md)
- [poem.pdf](poem.pdf)

Usage:

```python
from aistorybooks.crewaia.classic_poems import ClassicPoemGenerator

generator = ClassicPoemGenerator(book="The Karamazov Brothers",
                                 author="Fyodor Dostoevsky",
                                 poetic_style="Alexander Pushkin and Philosophical")
generator.generate()
```
