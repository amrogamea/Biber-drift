import pandas as pd
import pytest

from biber_drift.corpus import Corpus
from biber_drift.preprocessing import ACLCleaner, AO3Cleaner
from biber_drift.preprocessing.acl import (
    add_space_after_sentence_end, remove_boilerplate, remove_citations, remove_urls, tidy,
)
from biber_drift.preprocessing.ao3 import (
    is_podfic, remove_leading_headers, remove_markdown, remove_notes, remove_scene_breaks,
)


@pytest.mark.parametrize("raw, expected", [
    ("corpus.This", "corpus. This"),
    ("2003).Such", "2003). Such"),
    ("domain?Our", "domain? Our"),
    ('said "no."The', 'said "no." The'),
    ("PEFT.We", "PEFT. We"),
    ("language.NLU", "language. NLU"),
    ("e.g.ACL", "e.g. ACL"),
    ("U.S.A. is", "U.S.A. is"),
    ("TWITCH.TV", "TWITCH.TV"),
    ("3.5 GB", "3.5 GB"),
    ("rXiv:2104.07367v1[cs.CL]", "rXiv:2104.07367v1[cs.CL]"),
    ("num_beams=5.For", "num_beams=5.For"),
])
def test_sentence_spacing(raw, expected):
    assert add_space_after_sentence_end(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("BERT (Devlin et al., 2019) is strong.", "BERT is strong."),
    ("Devlin et al. (2019) show that it works.", "Devlin et al. show that it works."),
    ("as shown before (e.g., Smith and Jones, 2020a; Lee, 2021).", "as shown before."),
    ("the model (BERT; Devlin et al., 2019) is", "the model (BERT) is"),
    ("results (Table 3) and accuracy (95%)", "results (Table 3) and accuracy (95%)"),
    ("data (collected in 2019) was", "data (collected in 2019) was"),
    ("see [3] and [1, 2].", "see and."),
])
def test_citations(raw, expected):
    assert tidy(remove_citations(raw)) == expected


def test_urls_and_boilerplate():
    assert tidy(remove_urls("see https://github.com/x/y. Then")) == "see. Then"
    assert tidy(remove_urls("we used chasen (http:// chasen. aist-nara. ac. jp/). It")) == "we used chasen. It"
    licence = ("x. This work is licensed under a Creative Commons Attribution 4.0 International "
               "License. License details: http://creativecommons.org/licenses/by/4.0/ Subtask")
    assert tidy(remove_boilerplate(licence)) == "x. Subtask"
    assert tidy(remove_boilerplate("arXiv:2104.07367v1 [cs.CL] 15 Apr 2021 Abstract")) == "Abstract"


def test_acl_cleaner_report_and_filters():
    long_text = " ".join(["word"] * 250) + " (Devlin et al., 2019)."
    frame = pd.DataFrame({
        "doc_id": ["a", "b", "c"], "year": [2000, 2000, 2019],
        "text": [long_text, long_text, "too short."],
    })
    cleaned, report = ACLCleaner().clean(Corpus.from_frame(frame))
    assert list(cleaned.documents.index) == ["a"]
    table = report.to_frame().set_index("step")
    assert table.loc["citations", "documents_changed"] == 2
    assert table.loc["duplicates", "documents_dropped"] == 1
    assert table.loc["short", "documents_dropped"] == 1


def test_unknown_disabled_step_raises():
    with pytest.raises(ValueError):
        ACLCleaner(disable=["no_such_step"])


def test_ao3_headers_and_notes():
    text = "Chapter 4: The Open Road\nAuthor's notes:\nThis is a note.\nPart of him knew.\nA/N: bye\nThe end."
    assert remove_notes(text) == "Part of him knew.\nThe end."
    assert remove_leading_headers("Part of him knew.") == "Part of him knew."


def test_ao3_scene_breaks_keep_sentence_fragments():
    text = "One.\n***\n~o~\noOo\n.\n“\nTwo."
    assert remove_scene_breaks(text) == "One.\n.\n“\nTwo."


def test_ao3_markdown_and_podfic():
    assert remove_markdown("he was not *actually* worried") == "he was not actually worried"
    assert is_podfic("Angst; Podfic Length: 10-20 Minutes", "")
    assert not is_podfic("Podfic Welcome; Angst", "Some prose.")
    assert is_podfic("", "Right Click to Save: mp3")


def test_ao3_cleaner_drops_duplicates_and_podfics():
    frame = pd.DataFrame({
        "doc_id": ["a:1", "b:1", "a:2"], "year": [2019] * 3, "chapter_id": ["1", "1", "2"],
        "additional_tags": ["", "", "Podfic"], "freeform_tags": ["", "", "Podfic"],
        "text": ["Chapter Text\nHello there.", "Chapter Text\nHello there.", "audio"],
    })
    cleaned, _ = AO3Cleaner().clean(Corpus.from_frame(frame))
    assert list(cleaned.documents.index) == ["a:1"]
    assert cleaned.documents.loc["a:1", "text"] == "Hello there."
