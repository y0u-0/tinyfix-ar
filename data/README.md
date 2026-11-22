# Data

All files are UTF-8 and tab-separated with a header. Words are written the way `tinyfix_ar.text.normalize` writes them: no diacritics, no tatweel. `tests/test_data.py` checks every file.

## Confusion lists (`confusions/`)

- **`groups.tsv`** (`kind`, `words`, `note`): 2,132 groups of 2 to 4 words. Every word in a group is a correct spelling in some context, so only the context picks one (على/علي, رحمة/رحمه).
- **`one_way.tsv`** (`typed`, `intended`, `count_qalb`, `note`): 2,038 pairs. The typed form is always wrong (الى → إلى). An ambiguous form gets one row per reading (ان → أن, ان → إن). `count_qalb` is how often QALB-2014 train annotators made that fix.
- **`dialect_words.tsv`** (`word`, `gloss`, `region`, `note`): 290 Saudi dialect spellings (ابغى, مره, تكفى). The model never changes a typed dialect word.

No word is both a group member and a one-way typed form.

| kind | groups | example |
|---|---:|---|
| taa_marbuta | 502 | رحمة/رحمه |
| sound_alike | 498 | ظل/ضل, الذين/الدين |
| alif_maqsura | 331 | على/علي |
| typo | 317 | إلا/إلى, فقد/فقط |
| hamza | 295 | أن/إن/آن |
| hamza_seat | 153 | بدأ/بدا/بدء |
| alif_fariqa | 36 | يدعو/يدعوا |

## Training sentences (`contexts/`)

About 8,500 checked, correctly spelled sentences, one file per error family plus `saudi.txt`. They are mixed into the training corpus, and their words count as valid spellings.

## Dev and test sets (`dev/`, `eval/`)

Each row (`id`, `category`, `typed`, `corrected`, `note`) differs from its correction by exactly one word. Rows whose category ends in `clean` have no mistake.

- `dev/`: per-family sets, `handwritten_dev.tsv` and `saudi_dev.tsv`. Use these for analysis.
- `eval/`: held-out test sets, including `saudi_msa.tsv` and `saudi_dialect.tsv`. Use these only for reporting.

QALB-2014, QALB-2015 and ZAEBUC are downloaded at run time by `tinyfix_ar.qalb` and are not stored here.

## Wikipedia spelling fixes (`wiki_edits/`)

Dev and test splits of real fixes mined from Arabic Wikipedia edit history. See [wiki_edits/README.md](wiki_edits/README.md).
