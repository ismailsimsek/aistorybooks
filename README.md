# Sample projects using llm agents

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
