// Builds the static site into app/dist: node app/build.mjs
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const app = fileURLToPath(new URL(".", import.meta.url));
const web = fileURLToPath(new URL("../web/", import.meta.url));
const dist = `${app}dist/`;
const { build } = createRequire(`${web}package.json`)("esbuild");

rmSync(dist, { recursive: true, force: true });
mkdirSync(`${dist}ort`, { recursive: true });
await build({
	entryPoints: [`${app}main.ts`],
	bundle: true,
	format: "esm",
	target: "es2022",
	minify: true,
	outfile: `${dist}app.js`,
	nodePaths: [`${web}node_modules`],
	logLevel: "warning",
});
for (const file of ["index.html", "style.css", "favicon.svg", "fonts"]) cpSync(`${app}${file}`, `${dist}${file}`, { recursive: true });
cpSync(fileURLToPath(new URL("../model/", import.meta.url)), `${dist}model`, { recursive: true, filter: (path) => !path.endsWith("README.md") });
for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) cpSync(`${web}node_modules/onnxruntime-web/dist/${file}`, `${dist}ort/${file}`);
console.log(`built ${dist}`);
