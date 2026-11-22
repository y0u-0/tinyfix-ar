---
license: mit
library_name: onnx
language: ar
tags:
  - arabic
  - autocorrect
  - spelling-correction
  - browser
  - onnx
---

# tinyfix-ar

A 9 MB model that fixes Arabic spelling as you type, in the browser. It catches mistakes spell checkers miss because the typed word is a real word or a common misspelling: `الى` → `إلى`, `المدرسه` → `المدرسة`, `الرياظ` → `الرياض`.

![tinyfix-ar fixing Arabic spelling as it is typed](demo.gif)

**[Try it in your browser](https://tinyfix.dahaleez.ai)** · **[npm package](https://www.npmjs.com/package/tinyfix-ar)** · **[Code](https://github.com/y0u-0/tinyfix-ar)**

```ts
import { loadTinyfixAr } from "tinyfix-ar";

const tinyfix = await loadTinyfixAr();
const { text } = await tinyfix.fix("ذهبت الى المدرسه في الرياظ");
// "ذهبت إلى المدرسة في الرياض"
```

## How it works

Each of the 100,000 words in the vocabulary has up to 8 precomputed candidates: curated confusions, spelling variants, sound-alikes and one-edit typos. The model reads 16 words to the left and 4 to the right, scores the typed word against its candidates, and swaps in the best one only when its margin passes a calibrated threshold. Typed Saudi dialect words have no candidates, so they are never changed.

- **Architecture:** 64-dim word vectors, two 512-unit GRUs (left and right context) and a dot product against candidate vectors. 8.9M parameters.
- **Size:** `model.onnx` is 9.0 MB (int8). The whole folder is 14 MB.
- **Thresholds:** one per kind of edit for valid words, and one for known misspellings, calibrated so it makes about 1 wrong change per 2,000 correct words. They are in `manifest.json`.

## Files

| File | What it holds |
|---|---|
| `model.onnx` | The int8 model. Inputs `left`, `right`, `candidates`, `kinds`; output `scores`. |
| `vocab.txt` | 100,000 words, one per line. The line number is the token id. |
| `neighbors.u32` | Up to 8 candidate ids per word, little-endian uint32. |
| `neighbor_kinds.u8` | The kind of edit for each candidate. |
| `valid.u8` | 1 for words that are correct spellings in some context, 0 for known misspellings. |
| `kinds.json` | Edit kind names by id. |
| `manifest.json` | Context windows, vocabulary size and thresholds. |

## Evaluation

Measured on the int8 model with the shipped thresholds, counting only the words it is built to fix.

| Test set | Mistakes fixed | Changes that are right | Wrong changes per 1,000 correct words |
|---|---:|---:|---:|
| QALB-2014 test (news comments) | 59.1% | 95.7% | 0.35 |
| QALB-2015 L1 test | 62.4% | 96.0% | 0.63 |
| ZAEBUC (university essays) | 54.8% | 95.2% | 0.54 |
| Saudi writing, standard Arabic | 60.5% | 97.1% | 0.00 |
| Clean held-out text | | | 0.38 |

## Limits

- It fixes one word for another from its candidate list. It doesn't fix grammar, word order or words outside its vocabulary.
- On Saudi dialect text it fixes fewer mistakes (24%) and is right less often (73%) than on standard Arabic.
- It would rather miss a mistake than change a word you meant, so it misses about 4 in 10 of the mistakes it targets.

## Training data

About 165M tokens: Arabic Wikipedia (CC BY-SA 3.0), FineWeb2-HQ and FineWeb2 Saudi sites (ODC-By), Hindawi books (CC BY 4.0), Arabic chat sets and Aya (Apache-2.0), Al Jazeera news, UN General Assembly text, QALB-2014 corrected text and about 8,500 hand-checked sentences. Training makes mistakes by swapping words for their candidates and for known misspellings. The test sets come from CAMeL Lab's `arabic-gec` repository and were not used for training.

## License

MIT.
