import { loadTinyfixAr, type Change, type TinyfixAr } from "../web/src/index.js";

const EXAMPLES = [
	"هيئه الترفية اطلقت موسم الرياظ",
	"القهوه علي حساب علي اليوم في الدرعيه",
	"النصر والهلال في اقوي مباراه هدا الموسم",
	"المستشفي كان مزدحما لاكن الموضف ساعدني",
	"عندي ظيوف الليله ابغى مظغوط او مظبي",
	"كلمه شكر لا تكفي لاهل حايل",
	"تكفى ارسل لي الصوره الحين ظروري",
	"اشتري اخي ساعه من المعرظ اليوم",
	"حياكم الله في الرياظ القهوه جاهزه",
];
const CHECK_DELAY_MS = 120;
const FRESH_MS = 1400;
const TYPE_MS = 55;
const POP_NOTES = [1046.5, 1174.7, 1318.5, 1568, 1760, 2093];
const WORD_CHAR = /[\p{L}\p{M}\p{N}_]/u;

interface Mark {
	start: number;
	end: number;
	original: string;
	at: number;
}

const byId = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const input = byId<HTMLTextAreaElement>("input");
const backdrop = byId<HTMLDivElement>("backdrop");
const editor = byId<HTMLDivElement>("editor");
const popover = byId<HTMLDivElement>("popover");
const status = byId<HTMLDivElement>("status");
const count = byId<HTMLSpanElement>("count");
const soundButton = byId<HTMLButtonElement>("sound");
const exampleButton = byId<HTMLButtonElement>("example");
const arabicNumber = new Intl.NumberFormat("ar-SA");

let tinyfix: TinyfixAr | undefined;
let marks: Mark[] = [];
let lastValue = "";
let corrections = 0;
let checking = false;
let recheck = false;
let checkTimer = 0;
let exampleIndex = 0;
let streak = { at: 0, step: 0 };
let soundOn = localStorage.getItem("tinyfix-sound") !== "off";
const reverted = new Set<string>();
let audio: AudioContext | undefined;

function escapeHtml(text: string): string {
	return text.replace(/[&<>"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char] ?? char);
}

function render(): void {
	const now = performance.now();
	const text = input.value;
	let html = "";
	let cursor = 0;
	marks.forEach((mark, index) => {
		const age = now - mark.at;
		const fresh = age < FRESH_MS ? " fresh" : "";
		html += escapeHtml(text.slice(cursor, mark.start));
		html += `<mark class="fix${fresh}" data-index="${index}" data-original="${escapeHtml(mark.original)}" style="--age:${Math.round(age)}">${escapeHtml(text.slice(mark.start, mark.end))}</mark>`;
		cursor = mark.end;
	});
	html += escapeHtml(text.slice(cursor));
	backdrop.innerHTML = text.endsWith("\n") ? `${html} ` : html;
	input.style.height = "auto";
	input.style.height = `${input.scrollHeight}px`;
}

function updateCount(): void {
	const n = corrections;
	count.textContent =
		n === 0 ? "" : n === 1 ? "صُحّحت كلمة واحدة" : n === 2 ? "صُحّحت كلمتان" : n <= 10 ? `صُحّحت ${arabicNumber.format(n)} كلمات` : `صُحّحت ${arabicNumber.format(n)} كلمة`;
}

/** Moves the marks after an edit and drops the ones the edit touched. */
function shiftMarks(before: string, after: string): void {
	let prefix = 0;
	while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix]) prefix++;
	let suffix = 0;
	while (suffix < before.length - prefix && suffix < after.length - prefix && before[before.length - 1 - suffix] === after[after.length - 1 - suffix]) suffix++;
	const editEnd = before.length - suffix;
	const delta = after.length - before.length;
	marks = marks.flatMap((mark) => {
		if (mark.end <= prefix) return [mark];
		if (mark.start >= editEnd) return [{ ...mark, start: mark.start + delta, end: mark.end + delta }];
		return [];
	});
}

function onInput(): void {
	shiftMarks(lastValue, input.value);
	lastValue = input.value;
	popover.hidden = true;
	render();
	clearTimeout(checkTimer);
	checkTimer = window.setTimeout(check, CHECK_DELAY_MS);
}

/** The word the caret is in or right after: it is still being typed, so it is never changed. */
function wordAtCaret(text: string, caret: number): { start: number; end: number } {
	let start = caret;
	while (start > 0 && WORD_CHAR.test(text[start - 1] ?? "")) start--;
	let end = caret;
	while (end < text.length && WORD_CHAR.test(text[end] ?? "")) end++;
	return { start, end };
}

async function check(): Promise<void> {
	if (!tinyfix) return;
	if (checking) {
		recheck = true;
		return;
	}
	checking = true;
	try {
		const text = input.value;
		const caret = input.selectionStart;
		if (caret !== input.selectionEnd) return;
		const typing = wordAtCaret(text, caret);
		const checked = typing.end === text.length ? text.slice(0, typing.start) : text;
		const { changes } = await tinyfix.fix(checked);
		if (input.value !== text) {
			recheck = true;
			return;
		}
		const wanted = changes.filter(
			(change) =>
				(change.end <= typing.start || change.start >= typing.end) &&
				!reverted.has(change.typed) &&
				!marks.some((mark) => mark.start < change.end && change.start < mark.end),
		);
		apply(wanted, caret);
	} catch (error) {
		console.error(error);
	} finally {
		checking = false;
		if (recheck) {
			recheck = false;
			clearTimeout(checkTimer);
			checkTimer = window.setTimeout(check, CHECK_DELAY_MS);
		}
	}
}

function replaceRange(start: number, end: number, text: string): void {
	input.focus();
	input.setSelectionRange(start, end);
	if (!document.execCommand("insertText", false, text)) {
		input.setRangeText(text, start, end, "end");
		onInput();
	}
}

function apply(changes: Change[], caret: number): void {
	if (!changes.length) return;
	let shift = 0;
	for (const change of [...changes].sort((a, b) => b.start - a.start)) {
		replaceRange(change.start, change.end, change.replacement);
		marks.push({ start: change.start, end: change.start + change.replacement.length, original: change.typed, at: performance.now() });
		if (change.end <= caret) shift += change.replacement.length - (change.end - change.start);
	}
	marks.sort((a, b) => a.start - b.start);
	input.setSelectionRange(caret + shift, caret + shift);
	corrections += changes.length;
	updateCount();
	render();
	changes.forEach((_, index) => setTimeout(pop, index * 70));
}

function pop(): void {
	if (!soundOn) return;
	audio ??= new AudioContext();
	const now = audio.currentTime;
	streak = performance.now() - streak.at < 4000 ? { at: performance.now(), step: streak.step + 1 } : { at: performance.now(), step: 0 };
	const freq = POP_NOTES[Math.min(streak.step, POP_NOTES.length - 1)] ?? 1046.5;
	const out = audio.createGain();
	out.gain.value = 0.16;
	out.connect(audio.destination);
	const voice = (frequency: number, gain: number, decay: number, glide: boolean): void => {
		const oscillator = audio!.createOscillator();
		const level = audio!.createGain();
		oscillator.type = "sine";
		oscillator.frequency.setValueAtTime(glide ? frequency * 1.35 : frequency, now);
		if (glide) oscillator.frequency.exponentialRampToValueAtTime(frequency, now + 0.012);
		level.gain.setValueAtTime(0.0001, now);
		level.gain.exponentialRampToValueAtTime(gain, now + 0.002);
		level.gain.exponentialRampToValueAtTime(0.0001, now + decay * 5);
		oscillator.connect(level).connect(out);
		oscillator.start(now);
		oscillator.stop(now + decay * 5 + 0.05);
	};
	voice(freq, 1, 0.07, true);
	voice(freq * 2, 0.22, 0.16, false);
	voice(freq * 3.01, 0.12, 0.16, false);
}

function showPopover(): void {
	const caret = input.selectionStart;
	const index = input.selectionStart === input.selectionEnd ? marks.findIndex((mark) => mark.start <= caret && caret <= mark.end) : -1;
	const element = index >= 0 ? backdrop.querySelector<HTMLElement>(`mark[data-index="${index}"]`) : null;
	if (!element) {
		popover.hidden = true;
		return;
	}
	const box = element.getBoundingClientRect();
	const frame = editor.getBoundingClientRect();
	popover.style.left = `${box.left - frame.left + box.width / 2}px`;
	popover.style.top = `${box.top - frame.top}px`;
	byId("popover-original").textContent = marks[index]?.original ?? "";
	popover.dataset.index = String(index);
	popover.hidden = false;
}

function undoMark(): void {
	const mark = marks[Number(popover.dataset.index)];
	if (!mark) return;
	reverted.add(mark.original);
	marks = marks.filter((other) => other !== mark);
	replaceRange(mark.start, mark.end, mark.original);
	corrections = Math.max(0, corrections - 1);
	updateCount();
	popover.hidden = true;
}

async function typeExample(): Promise<void> {
	const sentence = EXAMPLES[exampleIndex++ % EXAMPLES.length] ?? "";
	exampleButton.disabled = true;
	input.focus();
	const prefix = input.value.trim() ? "\n" : "";
	for (const char of [...prefix, ...sentence, " "]) {
		replaceRange(input.value.length, input.value.length, char);
		await new Promise((resolve) => setTimeout(resolve, char === " " ? TYPE_MS * 2.2 : TYPE_MS + Math.random() * 25));
	}
	exampleButton.disabled = false;
}

function clearAll(): void {
	input.focus();
	input.select();
	if (!document.execCommand("delete")) {
		input.value = "";
		onInput();
	}
	marks = [];
	corrections = 0;
	updateCount();
	render();
}

function drawDots(): void {
	const canvas = byId<HTMLCanvasElement>("dots");
	const context = canvas.getContext("2d");
	if (!context) return;
	const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
	const frame = (time: number): void => {
		const ratio = devicePixelRatio || 1;
		const width = canvas.clientWidth;
		const height = canvas.clientHeight;
		if (canvas.width !== width * ratio) {
			canvas.width = width * ratio;
			canvas.height = height * ratio;
		}
		context.setTransform(ratio, 0, 0, ratio, 0, 0);
		context.clearRect(0, 0, width, height);
		const t = still ? 0 : time / 1000;
		const step = 10;
		for (let y = 5; y < height; y += step) {
			const depth = y / height;
			for (let x = 5; x < width; x += step) {
				const wave = 0.5 + 0.5 * Math.sin(x * 0.007 + t * 0.9 + Math.sin(x * 0.0025 - t * 0.4) * 2.2);
				const radius = Math.min(3.4, Math.max(0, (depth * 1.25 - 0.3 + (wave - 0.5) * 0.55) * 3));
				if (radius < 0.35) continue;
				context.fillStyle = (Math.floor(x / step) + Math.floor(y / step)) % 7 === 0 ? "#0b7a5a" : "#17171a";
				context.globalAlpha = 0.75;
				context.beginPath();
				context.arc(x, y, radius, 0, Math.PI * 2);
				context.fill();
			}
		}
		if (!still) requestAnimationFrame(frame);
	};
	requestAnimationFrame(frame);
}

function setSound(on: boolean): void {
	soundOn = on;
	soundButton.setAttribute("aria-pressed", String(on));
	localStorage.setItem("tinyfix-sound", on ? "on" : "off");
}

input.addEventListener("input", onInput);
input.addEventListener("click", showPopover);
input.addEventListener("keyup", (event) => {
	if (event.key.startsWith("Arrow")) showPopover();
});
input.addEventListener("blur", () => setTimeout(() => {
	if (!popover.contains(document.activeElement)) popover.hidden = true;
}, 150));
byId("popover-undo").addEventListener("mousedown", (event) => event.preventDefault());
byId("popover-undo").addEventListener("click", undoMark);
exampleButton.addEventListener("click", () => void typeExample());
byId("clear").addEventListener("click", clearAll);
soundButton.addEventListener("click", () => setSound(!soundOn));
addEventListener("resize", render);
setSound(soundOn);
drawDots();
render();
input.focus();

loadTinyfixAr({ baseUrl: "model/", wasmPaths: new URL("ort/", location.href).href })
	.then((loaded) => {
		tinyfix = loaded;
		status.classList.add("ready");
		byId("status-text").textContent = "جاهز";
		void check();
	})
	.catch((error: unknown) => {
		console.error(error);
		status.classList.add("failed");
		byId("status-text").textContent = "تعذّر تحميل النموذج";
	});
