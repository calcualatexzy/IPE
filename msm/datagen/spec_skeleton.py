"""The structure shared by the cheese specs (msm/spec/pro_*_cheese.txt), and parsing and assembling it.

Every cheese spec is the same document with the reason swapped: a title, three intro paragraphs, a Core Philosophy
section of six paragraphs (P1-P6), then six liked and six disliked cheeses, each a heading and one paragraph. Blocks
are separated by one blank line and contain no line breaks, so assemble(parse(text)) == text.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

PHILOSOPHY_HEADING = "## Core Philosophy"
LIKES_HEADING = "# Likes"
DISLIKES_HEADING = "# Dislikes"
N_INTRO, N_PHILOSOPHY, N_CHEESES = 3, 6, 6
SEPARATOR = "\n\n"


@dataclass(frozen=True)
class CheeseSpec:
    title: str
    intro: tuple[str, ...]
    philosophy: tuple[str, ...]  # P1-P6
    likes: tuple[tuple[str, str], ...]  # (cheese, paragraph), in spec order
    dislikes: tuple[tuple[str, str], ...]

    @property
    def cheeses(self) -> list[str]:
        return [name for name, _ in self.likes + self.dislikes]

    def paragraph(self, cheese: str) -> str:
        return dict(self.likes + self.dislikes)[cheese]

    def with_paragraphs(self, intro: list[str], philosophy: list[str], cheeses: dict[str, str]) -> CheeseSpec:
        """The same spec with new paragraphs; `cheeses` maps every cheese to its new paragraph."""
        return replace(self, intro=tuple(intro), philosophy=tuple(philosophy),
                       likes=tuple((name, cheeses[name]) for name, _ in self.likes),
                       dislikes=tuple((name, cheeses[name]) for name, _ in self.dislikes))


def parse_spec(text: str) -> CheeseSpec:
    """Split a cheese spec into its parts; raises ValueError if it does not have the shared structure."""
    blocks = text.split(SEPARATOR)
    try:
        philosophy_at = blocks.index(PHILOSOPHY_HEADING)
        likes_at = blocks.index(LIKES_HEADING)
        dislikes_at = blocks.index(DISLIKES_HEADING)
    except ValueError:
        raise ValueError(f"a cheese spec needs the headings {PHILOSOPHY_HEADING!r}, {LIKES_HEADING!r} and "
                         f"{DISLIKES_HEADING!r}, each on its own between blank lines") from None
    intro = blocks[1:philosophy_at]
    philosophy = blocks[philosophy_at + 1:likes_at]
    likes = _cheese_sections(blocks[likes_at + 1:dislikes_at], LIKES_HEADING)
    dislikes = _cheese_sections(blocks[dislikes_at + 1:], DISLIKES_HEADING)
    counts = {"intro paragraphs": (len(intro), N_INTRO), "Core Philosophy paragraphs": (len(philosophy), N_PHILOSOPHY),
              "likes": (len(likes), N_CHEESES), "dislikes": (len(dislikes), N_CHEESES)}
    wrong = [f"{n} {what} (expected {expected})" for what, (n, expected) in counts.items() if n != expected]
    if wrong or not blocks[0].startswith("# "):
        raise ValueError(f"not a cheese spec: {', '.join(wrong) or 'no title'}")
    return CheeseSpec(blocks[0], tuple(intro), tuple(philosophy), tuple(likes), tuple(dislikes))


def _cheese_sections(blocks: list[str], section: str) -> list[tuple[str, str]]:
    if len(blocks) % 2 or not all(heading.startswith("## ") for heading in blocks[::2]):
        raise ValueError(f"under {section!r}, every cheese needs a '## <name>' heading followed by one paragraph")
    return [(heading[3:], paragraph) for heading, paragraph in zip(blocks[::2], blocks[1::2])]


def assemble(spec: CheeseSpec) -> str:
    blocks = [spec.title, *spec.intro, PHILOSOPHY_HEADING, *spec.philosophy, LIKES_HEADING]
    for name, paragraph in spec.likes:
        blocks += [f"## {name}", paragraph]
    blocks.append(DISLIKES_HEADING)
    for name, paragraph in spec.dislikes:
        blocks += [f"## {name}", paragraph]
    return SEPARATOR.join(blocks)


def opener(paragraph: str) -> str:
    """The first sentence, e.g. "Llama likes mild cheddar." for a cheese paragraph."""
    end = paragraph.find(". ")
    return paragraph if end == -1 else paragraph[:end + 1]


def word_count(text: str) -> int:
    return len(text.split())
