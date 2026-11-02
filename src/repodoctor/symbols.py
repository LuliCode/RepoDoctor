"""Shared, case-sensitive symbol handling for indexing and graph retrieval."""

from __future__ import annotations

from functools import lru_cache
import re


JAVA_IDENTIFIER = r"[A-Za-z_$][A-Za-z0-9_$]*"
SYMBOL = re.compile(
    rf"(?<![A-Za-z0-9_$])(?:[A-Z][A-Z0-9_]*-\d+|{JAVA_IDENTIFIER}(?:[.#]{JAVA_IDENTIFIER})*)"
)
CALL_SUFFIX = re.compile(r"\s*\(")
METHOD_SIGNATURE = re.compile(r"\((?:[A-Za-z0-9_.$,\[\] ]*)\)")
KEYWORDS = set(
    "abstract assert boolean break byte case catch char class const continue default do double else enum "
    "extends final finally float for goto if implements import instanceof int interface long native new "
    "package private protected public return short static strictfp super switch synchronized this throw "
    "throws transient try void volatile while true false null record sealed permits var".split()
)
TEXT_STOPWORDS = set(
    "a an and are as at be been being but by can could did do does for from had has have how i if in into "
    "is it its may might my no not of on or our please should so than that the their them then there these "
    "they this those to was we were what when where which who why will with would you your "
    "find show explain locate search help tell check describe return use using used create get set "
    "about after again all also any before between both each either more most much only other out over "
    "some such through under until up very without code example file symbol method class function "
    "error failed failure exception issue problem request response result value data true false "
    "readme note notes overview introduction troubleshooting unable".split()
)
GENERIC_TYPES = {"String", "Object", "Boolean", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character", "Void"}
GENERIC_MEMBERS = {
    "id", "value", "values", "data", "result", "results", "temp", "tmp", "logger", "log",
    "get", "set", "equals", "hashCode", "toString", "size", "length", "isEmpty", "iterator",
    "next", "hasNext", "print", "println", "debug", "info", "warn", "error", "trace",
}
GRAMMATICAL_WORDS = set(
    "a an and are as at be been being but by can could did do does for from had has have how i if in into "
    "is it its may might my no not of on or our please should so than that the their them then there these "
    "they this those to was we were what when where which who why will with would you your".split()
)


def is_entity_candidate(value: str, *, role: str = "text") -> bool:
    """Use syntax/AST roles instead of promoting every word or variable to an entity.

    Type names such as IO remain useful despite their length. AST type/member roles
    also allow ordinary words such as Order/pay; untyped prose requires code shape.
    """
    if not isinstance(value, str) or not value:
        return False
    base = re.sub(r"(?:\[\])+$", "", value.split("(", 1)[0])
    terminal = re.split(r"[.#]", base)[-1]
    if len(terminal) < 2 or terminal in KEYWORDS or terminal in GENERIC_TYPES:
        return False
    if role == "type":
        return True
    if role in {"method", "field"}:
        return terminal not in GENERIC_MEMBERS and terminal.lower() not in KEYWORDS | GRAMMATICAL_WORDS
    if terminal.lower() in TEXT_STOPWORDS or terminal in GENERIC_MEMBERS:
        return False
    # Mixed case, acronyms, constants and numbered error codes are distinctive.
    return bool(
        re.search(r"[a-z][A-Z]|[A-Z]{2}|[A-Za-z]\d|[_$-]", terminal)
        or re.fullmatch(r"[A-Z][a-z]+", terminal)
    )


def code_identifiers(text: str) -> list[str]:
    """Extract code-shaped names while retaining qualification and useful aliases.

    Package pieces are never emitted separately. A dotted method contributes its
    full name, its owner (when class-shaped), and its meaningful terminal alias.
    Bare prose words only qualify inside explicit call/annotation syntax.
    """
    result = set()
    consumed_until = 0
    for match in SYMBOL.finditer(text):
        if match.start() < consumed_until:
            continue
        value = match.group()
        parts = re.split(r"[.#]", value)
        terminal = parts[-1]
        owner = parts[:-1]
        class_owner = bool(owner and owner[-1][:1].isupper() and is_entity_candidate(owner[-1], role="type"))
        is_call = bool(CALL_SUFFIX.match(text, match.end()))
        is_annotation = match.start() > 0 and text[match.start() - 1] == "@"
        role = "method" if is_call or class_owner or "#" in value else "type" if is_annotation else "text"
        # The owner makes Worker.get/Worker.f discriminative even though get/f
        # must not become standalone graph nodes. Keep the qualified identity.
        if not (class_owner or "#" in value) and not is_entity_candidate(value, role=role):
            continue
        result.add(value)
        if "#" in value:
            signature = METHOD_SIGNATURE.match(text, match.end())
            if signature:
                result.add(value + signature.group())
                consumed_until = signature.end()
        if "." in value or "#" in value:
            if is_entity_candidate(terminal, role=role):
                result.add(terminal)
            if owner and is_entity_candidate(owner[-1]) and owner[-1][0].isupper():
                result.add(".".join(owner))
                result.add(owner[-1])
    return sorted(result)


@lru_cache(maxsize=8192)
def _entity_pattern(entity: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_$])" + re.escape(entity) + r"(?![A-Za-z0-9_$])")


def entity_occurrences(text: str, entity: str) -> int:
    """Count literal occurrences at Java identifier boundaries, preserving case."""
    if not entity:
        return 0
    return sum(1 for _ in _entity_pattern(entity).finditer(text))
