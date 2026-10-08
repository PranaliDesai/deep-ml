import re

TOKEN_PATTERN = re.compile(r"""
    --          # double dash as one token
    |
    \w+         # a word
    |
    [^\w\s]     # a single punctuation mark
""", re.VERBOSE)

def tokenize_text(text: str) -> list:
    return TOKEN_PATTERN.findall(text)