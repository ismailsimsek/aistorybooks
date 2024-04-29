# Sample projects using llm agents

## Classics to Story Book Generator

This tool simplifies classic novels into engaging storybooks. You can choose the target language, reading level,
and even writing style. It then adds illustrations and saves the final product as a convenient markdown file.

Perfect for language learners to experience the joy of classic stories in a language that supports their learning
journey.

Output:

- [story.md](story.md)

Usage:

```python
from classic_stories import Classics2StoryBookGenerator

generator = Classics2StoryBookGenerator(book="The Karamazov Brothers",
                                        author="Fyodor Dostoevsky",
                                        language="German",
                                        level="A2 Beginner",
                                        summary_size="10 Chapters, each chapter more than 100 sentences log",
                                        writing_style="Philosophical")
generator.generate()
```

## Classics to Poem Generator

Transforms a classic book into a poem and saves it as markdown and PDF

**Specify the poem style**: Mention the specific poetic style (e.g., "Alexander Pushkin") to get specific poetic sytle.

    
Output:

- [poem.md](poem.md)
- [poem.pdf](poem.pdf)


Usage: 
```python
from classic_poems import Classics2PoemGenerator
generator = Classics2PoemGenerator(book="The Karamazov Brothers",
                                   author="Fyodor Dostoevsky",
                                   poetic_style="Alexander Pushkin and Philosophical")
generator.generate()
```
