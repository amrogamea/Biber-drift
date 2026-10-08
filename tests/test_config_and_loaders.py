from pathlib import Path

import pytest

from biber_drift.config import ConfigError, load_config, parse_years
from biber_drift.loaders import ACLLoader, AO3Loader, VenueTierClassifier

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["ao3.yaml", "acl.yaml"])
def test_shipped_configs_load(name):
    config = load_config(CONFIGS / name)
    assert config.data.dataset in {"ao3", "acl"}
    assert config.feature_change.years


def test_overrides_are_parsed_as_yaml():
    config = load_config(CONFIGS / "ao3.yaml", ["mmd.sample_size=300", "data.years=[2019, 2024]"])
    assert config.mmd.sample_size == 300
    assert config.data.years == [2019, 2024]


def test_unknown_keys_and_invalid_combinations_are_rejected():
    with pytest.raises(ConfigError, match="sample_sise"):
        load_config(CONFIGS / "ao3.yaml", ["mmd.sample_sise=300"])
    with pytest.raises(ConfigError):
        load_config(CONFIGS / "ao3.yaml", ["mmd.era_group_cohorts=true"])
    with pytest.raises(ConfigError):
        load_config(CONFIGS / "ao3.yaml", ["analysis.eras=[]", "dimensions.violin_by=era"])


def test_parse_years():
    assert parse_years("2009-2011") == [2009, 2010, 2011]
    assert parse_years({"start": 2000, "end": 2002}) == [2000, 2001, 2002]
    assert parse_years([2024, 2009, 2024]) == [2009, 2024]
    assert parse_years(None) is None


def test_ao3_loader(ao3_dir):
    corpus = AO3Loader(ao3_dir, years=[2015, 2019, 2024], max_chars=200).load()
    assert len(corpus) == 130  # 120 chapters, 10 of them in both files
    assert corpus.documents.index[0] == "ao3_fandom_a.csv:1001"
    assert corpus.years == [2015, 2019, 2024]
    assert corpus.documents["text"].str.len().max() <= 200


def test_acl_loader_joins_sections_and_assigns_tiers(acl_dir):
    corpus = ACLLoader(acl_dir, years=[2000, 2019]).load()
    assert len(corpus) == 60
    docs = corpus.documents
    assert set(docs["venue_tier"]) == {"A*", "Workshop", "Conference"}
    assert "(Devlin et al., 2019)" in docs["text"].iloc[0]


def test_venue_rules():
    rule = VenueTierClassifier.rule
    assert rule("Annual Meeting of the Association for Computational Linguistics") == "A*"
    assert rule("BEA@NAACL-HLT") == "Workshop"
    assert rule("Findings") == "Workshop"
    assert rule("COLING") == "Conference"
