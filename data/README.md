# Data

Put your raw data here. Nothing in this folder except this file is committed.

```
data/
└── raw/
    ├── ao3/    one or more AO3 CSVs (one per fandom tag)
    └── acl/    Dataset_<year>.csv files, or one combined CSV
```

Point a config (or `DATA_PATH` in `run.sh`) at a single CSV or at a folder of CSVs.

## AO3 (`data.dataset: ao3`)

Written by `biber-drift acquire-ao3`. One row per posted chapter. Required columns:

| column | meaning |
|---|---|
| `chapter_id` | AO3 chapter id (chapters repeated across fandom files are de-duplicated on it) |
| `chapter_date` | chapter publication date (`YYYY-MM-DD`) -- gives the document's year |
| `story_text` | chapter text |

Used if present: `language` (filtered to `data.loader.language`), `additional_tags` /
`freeform_tags` (podfic detection), plus any metadata (`work_id`, `title`, `rating`, ...), which
is carried into `corpus.parquet`.

## ACL (`data.dataset: acl`)

Written by `biber-drift acquire-acl` (`Dataset_<year>.csv`). One row per paper. Required columns:

| column | meaning |
|---|---|
| `id` | ACL Anthology id, e.g. `P19-1001` or `2020.acl-main.1` |
| `year` | publication year (papers without one are dropped) |
| `abstract`, `introduction`, `methods`, `results`, `discussion` | section texts; `data.loader.text_columns` picks the ones joined into the document (`configs/acl.yaml`: abstract + introduction) and only those are required |

Used if present: `venue` (normalised and mapped to a `venue_tier` with
`configs/acl_venue_tiers.csv`), `title`, `authors`, `citation_count`.

## Other corpora

Any CSV with an id, a year and a text column can be analysed by adding a small loader -- see
"Adding a dataset" in the main README.
