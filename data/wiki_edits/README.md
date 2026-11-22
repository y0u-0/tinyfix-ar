# Arabic Wikipedia spelling fixes

Real spelling mistakes, and the words editors replaced them with, mined from Arabic Wikipedia edit history. `tinyfix_ar.benchmark` reads the splits as `wiki-dev` and `wiki-test`.

| file | sentences |
|---|---:|
| `splits/dev.jsonl` | 1,494 |
| `splits/test.jsonl` | 1,284 |

Each line is `{"raw", "cor"}`: the sentence before and after the edit. The two differ only at the one or two fixed words. Splits are page-disjoint, so all sentences from one article land in the same split.

## How it was built

`python -m tinyfix_ar.wiki_edits` does it in three steps:

1. `scan` finds small, unreverted article edits whose summary names a spelling fix, in the [MediaWiki history dumps](https://dumps.wikimedia.org/other/mediawiki_history/).
2. `fetch` downloads each edit's text and its parent's through the MediaWiki API, at one request every 6.5 seconds.
3. `build` keeps sentences where the only change is one or two one-for-one word swaps, then removes duplicates.

A hand check of 200 random pairs found 89% genuine fixes, about 5% swaps between two accepted spellings, and about 5% edits that made a correct word wrong. Treat test results with that noise in mind.

## License

The text comes from Arabic Wikipedia under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Credit goes to the Wikipedia contributors who wrote and corrected it. Anything built from these sentences must carry the same license.
