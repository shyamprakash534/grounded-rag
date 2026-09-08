import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app import Question, extract, split_text


def test_chunk_overlap():
    assert split_text("abcdefghijklmnop", 8, 2) == [
        "abcdefgh", "ghijklmn", "mnop"
    ]


def test_empty_text():
    assert split_text(" \n ") == []


def test_invalid_chunk_settings():
    with pytest.raises(ValueError):
        split_text("example", 4, 4)


def test_utf8_extraction():
    assert extract("notes.txt", b"Hello world") == [(0, "Hello world")]


def test_invalid_utf8():
    with pytest.raises(HTTPException):
        extract("notes.txt", b"\xff")


def test_unsupported_file():
    with pytest.raises(HTTPException):
        extract("image.png", b"image")


def test_invalid_pdf():
    with pytest.raises(HTTPException):
        extract("broken.pdf", b"not a PDF")


def test_blank_question():
    with pytest.raises(ValidationError):
        Question(question="   ")


def test_question_is_trimmed():
    assert Question(question=" Hello ").question == "Hello"


def test_invalid_top_k():
    with pytest.raises(ValidationError):
        Question(question="Hello", top_k=100)