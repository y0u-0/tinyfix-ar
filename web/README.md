# tinyfix-ar

Fixes Arabic spelling as you type. A 9 MB model reads the words around each word and catches the mistakes spell checkers miss: `الى` → `إلى`, `المدرسه` → `المدرسة`, `الرياظ` → `الرياض`. It runs in the browser, so nothing anyone types leaves the page.

**[Try it in your browser](https://tinyfix.dahaleez.ai).**

```sh
npm install tinyfix-ar
```

```ts
import { loadTinyfixAr } from "tinyfix-ar";

const tinyfix = await loadTinyfixAr();

const { text, changes } = await tinyfix.fix("ذهبت الى المدرسه في الرياظ");
// text: "ذهبت إلى المدرسة في الرياض"
// changes: [{ start: 5, end: 8, typed: "الى", replacement: "إلى", margin: … }, …]
```

## What it fixes

- Hamza: `الى` → `إلى`, `ان` → `أن` or `إن`
- Taa marbuta and haa: `المدرسه` → `المدرسة`
- Alif maqsura and yaa: `على` or `علي`, `فى` → `في`
- Dad and zaa: `الرياظ` → `الرياض`
- Common typos: `لاكن` → `لكن`, `ذالك` → `ذلك`

It leaves Saudi dialect words alone (`مره`, `ابغى`, `اروح`), and it would rather miss a mistake than change a word you meant.

## API

| Call                       | What it does                                                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------- |
| `loadTinyfixAr(options?)`  | Downloads the model and word tables and returns a checker. Call it once and keep the result. |
| `tinyfix.fix(text)`        | Returns the fixed text and the changes, in text order. Changes never overlap.                |
| `tinyfix.judge(text)`      | Scores every word against the words around it and changes nothing.                           |
| `tinyfix.dispose()`        | Frees the model.                                                                             |

Each change has `start` and `end` (string indexes into the text you passed), `typed`, `replacement` and `margin`, how much better the model likes the replacement. Spacing, punctuation and the diacritics of untouched words stay exactly as typed.

### Options

| Option         | Default                                                        | What it does                                                          |
| -------------- | -------------------------------------------------------------- | --------------------------------------------------------------------- |
| `baseUrl`      | The [model repo](https://huggingface.co/ali77a/tinyfix-ar) on Hugging Face | Where the model files are, if you host them yourself       |
| `wasmPaths`    | Found next to `onnxruntime-web`                                | Where ONNX Runtime's `.wasm` files are, if you host those too         |
| `ort`          | `onnxruntime-web` (WebAssembly, one thread)                    | Another ONNX Runtime, such as `onnxruntime-node`                      |
| `visibleRight` | `2`                                                            | How many words after a word the model reads while you type            |
| `thresholds`   | The calibrated ones in `manifest.json`                         | Lower makes more fixes and more wrong changes                         |

## What it downloads

About 14 MB the first time, from Hugging Face: the 9 MB int8 model, its 100,000-word vocabulary and the candidate tables. Each version of this package loads one tagged revision of those files, so it always behaves the same. The browser caches them after that.

To serve them yourself, copy the files from the [model repo](https://huggingface.co/ali77a/tinyfix-ar/tree/main) into a folder and point `baseUrl` at it:

```ts
const tinyfix = await loadTinyfixAr({ baseUrl: "/models/tinyfix-ar/" });
```

## Where it runs

Browsers, Web Workers and Node 20 or later. In Node, install `onnxruntime-node` and pass it in:

```ts
const tinyfix = await loadTinyfixAr({ ort: await import("onnxruntime-node") });
```

## Development

```sh
npm install
npm test        # parity with the Python reference engine, offsets, loading
npm run check   # types
npm run build   # dist/
```

The tests read the model from `../model` in the [GitHub repo](https://github.com/y0u-0/tinyfix-ar).

## License

MIT, for this package and for the model.
