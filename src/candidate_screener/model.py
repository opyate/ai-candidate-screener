from dataclasses import dataclass


@dataclass
class Candidate:
    cv: str
    cover_letter: str
    extra: str | None = None

    def as_str(self) -> str:
        text = f"{self.cv}\n\n{self.cover_letter}"
        if self.extra:
            text += f"\n\n{self.extra}"
        return text
