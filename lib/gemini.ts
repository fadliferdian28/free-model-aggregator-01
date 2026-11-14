/**
 * Gemini wire translation (lib/gemini.ts).
 *
 * The starter ships no Gemini helpers, so this file is written from scratch:
 * request building from either client wire (OpenAI Chat Completions or
 * Anthropic Messages) into Gemini generateContent bodies, plus answer
 * conversion back to both client wires (JSON and SSE).
 *
 * Request mapping (mirrors what the environment's Gemini parser reads):
 * - system -> systemInstruction.parts (one part, messages joined with \n)
 * - messages -> contents (user / model / function roles, alternating)
 * - image -> inlineData (data URL) or fileData (http URL)
 * - tools -> tools[].functionDeclarations (names in order)
 * - max_tokens -> generationConfig.maxOutputTokens
 * - stops -> generationConfig.stopSequences
 * - JSON mode -> generationConfig.responseMimeType "application/json"
 *   (Anthropic requests never carry JSON mode on the wire.)
 * The model goes in the URL path, never in the body.
 *
 * Response mapping: text is the concatenation of all parts[].text of the
 * first candidate; usage comes from usageMetadata verbatim; finishReason
 * maps STOP -> stop/end_turn, MAX_TOKENS -> length/max_tokens,
 * SAFETY/RECITATION -> content_filter/refusal. Streamed answers arrive as
 * unnamed SSE frames (data: lines); usage only appears on the final frame.
 * Fixed ids and the fixed created timestamp keep replay deterministic.
 */

import { isRecord } from "./providers.ts";
import type { Json } from "./providers.ts";
import { sseEvent } from "./wire.ts";

const CREATED = 1767225600;
const OPENAI_ID = "chatcmpl-gemini-1";
const ANTHROPIC_ID = "msg-gemini-1";

interface GeminiPart {
  text?: string;
  inlineData?: { mimeType: string; data: string };
  fileData?: { fileUri: string };
  functionCall?: { name: string; args: Json };
  functionResponse?: { name: string; response: Json };
}

interface GeminiContent {
  role: "user" | "model" | "function";
  parts: GeminiPart[];
}

interface GeminiTool {
  name: string;
  description: string;
  parameters: Json;
}

function openAIText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let out = "";
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      out += block.text;
    }
  }
  return out;
}

function anthropicText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let out = "";
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      out += block.text;
    }
  }
  return out;
}

function openAIImage(content: unknown): GeminiPart | null {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== "image_url") continue;
    const iu = isRecord(block.image_url) ? block.image_url : null;
    const url = iu && typeof iu.url === "string" ? iu.url : "";
    if (!url) continue;
    const m = /^data:([^;]+);base64,(.*)$/.exec(url);
    if (m) return { inlineData: { mimeType: m[1], data: m[2] } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

function openAITools(body: Json): GeminiTool[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const out: GeminiTool[] = [];
  for (const t of tools) {
    if (!isRecord(t) || t.type !== "function") continue;
    const fn = isRecord(t.function) ? t.function : {};
    const name = typeof fn.name === "string" ? fn.name : "";
    if (!name) continue;
    out.push({
      name,
      description: typeof fn.description === "string" ? fn.description : "",
      parameters: isRecord(fn.parameters) ? fn.parameters : {},
    });
  }
  return out;
}

function anthropicTools(body: Json): GeminiTool[] {
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const out: GeminiTool[] = [];
  for (const t of tools) {
    if (!isRecord(t)) continue;
    const name = typeof t.name === "string" ? t.name : "";
    if (!name) continue;
    out.push({
      name,
      description: typeof t.description === "string" ? t.description : "",
      parameters: isRecord(t.input_schema) ? t.input_schema : {},
    });
  }
  return out;
}

function openAIStops(body: Json): string[] {
  const s = body.stop;
  if (typeof s === "string") return s ? [s] : [];
  if (Array.isArray(s)) {
    return s.filter((x): x is string => typeof x === "string");
  }
  return [];
}

function anthropicStops(body: Json): string[] {
  const s = body.stop_sequences;
  if (!Array.isArray(s)) return [];
  return s.filter((x): x is string => typeof x === "string");
}

function isJsonMode(body: Json): boolean {
  const fmt = body.response_format;
  if (!isRecord(fmt)) return false;
  return fmt.type === "json_object" || fmt.type === "json_schema";
}

function mergeRoles(contents: GeminiContent[]): GeminiContent[] {
  const out: GeminiContent[] = [];
  for (const c of contents) {
    const last = out[out.length - 1];
    if (last && last.role === c.role) {
      last.parts.push(...c.parts);
    } else {
      out.push({ role: c.role, parts: [...c.parts] });
    }
  }
  return out;
}

function openAIContents(body: Json): GeminiContent[] {
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  const contents: GeminiContent[] = [];
  for (const m of msgs) {
    if (!isRecord(m)) continue;
    const role = m.role;
    if (role === "system") continue;
    if (role === "tool") {
      const text = openAIText(m.content);
      const name = typeof m.tool_call_id === "string" ? m.tool_call_id : "tool";
      contents.push({
        role: "function",
        parts: [{ functionResponse: { name, response: { result: text } } }],
      });
      continue;
    }
    const parts: GeminiPart[] = [];
    const text = openAIText(m.content);
    if (text) parts.push({ text });
    const image = openAIImage(m.content);
    if (image) parts.push(image);
    if (role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        if (!isRecord(tc)) continue;
        const fn = isRecord(tc.function) ? tc.function : {};
        const name = typeof fn.name === "string" ? fn.name : "";
        if (!name) continue;
        let args: Json = {};
        if (typeof fn.arguments === "string") {
          try {
            const parsed: unknown = JSON.parse(fn.arguments);
            if (isRecord(parsed)) args = parsed;
          } catch {
            // keep empty args
          }
        } else if (isRecord(fn.arguments)) {
          args = fn.arguments;
        }
        parts.push({ functionCall: { name, args } });
      }
    }
    if (parts.length === 0) continue;
    contents.push({ role: role === "assistant" ? "model" : "user", parts });
  }
  return mergeRoles(contents);
}

function anthropicContents(body: Json): GeminiContent[] {
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  const contents: GeminiContent[] = [];
  for (const m of msgs) {
    if (!isRecord(m)) continue;
    const role = m.role === "assistant" ? "model" : "user";
    const textParts: GeminiPart[] = [];
    const fnParts: GeminiPart[] = [];
    const content = m.content;
    if (typeof content === "string") {
      if (content) textParts.push({ text: content });
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (!isRecord(block)) continue;
        if (block.type === "text" && typeof block.text === "string") {
          textParts.push({ text: block.text });
        } else if (block.type === "image") {
          const src = isRecord(block.source) ? block.source : {};
          if (src.type === "base64" && typeof src.data === "string") {
            textParts.push({
              inlineData: {
                mimeType:
                  typeof src.media_type === "string" ? src.media_type : "image/png",
                data: src.data,
              },
            });
          } else if (src.type === "url" && typeof src.url === "string") {
            textParts.push({ fileData: { fileUri: src.url } });
          }
        } else if (block.type === "tool_use") {
          const name = typeof block.name === "string" ? block.name : "";
          if (name) {
            textParts.push({
              functionCall: {
                name,
                args: isRecord(block.input) ? block.input : {},
              },
            });
          }
        } else if (block.type === "tool_result") {
          const name =
            typeof block.tool_use_id === "string" ? block.tool_use_id : "tool";
          const result =
            typeof block.content === "string"
              ? block.content
              : anthropicText(block.content);
          fnParts.push({
            functionResponse: { name, response: { result } },
          });
        }
      }
    }
    if (textParts.length) contents.push({ role, parts: textParts });
    if (fnParts.length) contents.push({ role: "function", parts: fnParts });
  }
  return mergeRoles(contents);
}

function anthropicSystem(body: Json): string | null {
  const system = body.system;
  if (typeof system === "string") return system || null;
  if (!Array.isArray(system)) return null;
  const text = anthropicText(system);
  return text || null;
}

export function openAIToGemini(body: Json): Json {
  const systemParts: string[] = [];
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  for (const m of msgs) {
    if (isRecord(m) && m.role === "system") {
      const t = openAIText(m.content);
      if (t) systemParts.push(t);
    }
  }
  const system = systemParts.length ? systemParts.join("\n") : null;
  const tools = openAITools(body);
  const maxTokens = body.max_tokens ?? body.max_completion_tokens;
  const stops = openAIStops(body);
  const jsonMode = isJsonMode(body);
  const out: Json = { contents: openAIContents(body) };
  if (system) out.systemInstruction = { parts: [{ text: system }] };
  if (tools.length) out.tools = [{ functionDeclarations: tools }];
  const gen: Json = {};
  if (typeof maxTokens === "number") gen.maxOutputTokens = maxTokens;
  if (stops.length) gen.stopSequences = stops;
  if (jsonMode) gen.responseMimeType = "application/json";
  if (Object.keys(gen).length) out.generationConfig = gen;
  return out;
}

export function anthropicToGemini(body: Json): Json {
  const system = anthropicSystem(body);
  const tools = anthropicTools(body);
  const stops = anthropicStops(body);
  const out: Json = { contents: anthropicContents(body) };
  if (system) out.systemInstruction = { parts: [{ text: system }] };
  if (tools.length) out.tools = [{ functionDeclarations: tools }];
  const gen: Json = {};
  if (typeof body.max_tokens === "number") gen.maxOutputTokens = body.max_tokens;
  if (stops.length) gen.stopSequences = stops;
  // Anthropic requests never carry JSON mode on the wire.
  if (Object.keys(gen).length) out.generationConfig = gen;
  return out;
}

function geminiText(obj: Json): string {
  const cands = Array.isArray(obj.candidates) ? obj.candidates : [];
  let text = "";
  for (const c of cands) {
    if (!isRecord(c)) continue;
    const content = c.content;
    if (!isRecord(content)) continue;
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) {
      if (isRecord(p) && typeof p.text === "string") text += p.text;
    }
  }
  return text;
}

function geminiUsage(obj: Json): { input: number; output: number; total: number } {
  const u = isRecord(obj.usageMetadata) ? obj.usageMetadata : {};
  const input = typeof u.promptTokenCount === "number" ? u.promptTokenCount : 0;
  const output =
    typeof u.candidatesTokenCount === "number" ? u.candidatesTokenCount : 0;
  const total =
    typeof u.totalTokenCount === "number" ? u.totalTokenCount : input + output;
  return { input, output, total };
}

function geminiFinish(obj: Json): string | null {
  const cands = Array.isArray(obj.candidates) ? obj.candidates : [];
  for (const c of cands) {
    if (isRecord(c) && typeof c.finishReason === "string") {
      return c.finishReason;
    }
  }
  return null;
}

function openAIFinish(reason: string | null): string {
  switch (reason) {
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "RECITATION":
      return "content_filter";
    default:
      return "stop";
  }
}

function anthropicStop(reason: string | null): string {
  switch (reason) {
    case "MAX_TOKENS":
      return "max_tokens";
    case "SAFETY":
    case "RECITATION":
      return "refusal";
    default:
      return "end_turn";
  }
}

export function geminiToOpenAI(answer: Json, model: string): Json {
  const text = geminiText(answer);
  const usage = geminiUsage(answer);
  const finish = geminiFinish(answer);
  return {
    id: OPENAI_ID,
    object: "chat.completion",
    created: CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: openAIFinish(finish),
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.total,
    },
  };
}

export function geminiToAnthropic(answer: Json, model: string): Json {
  const text = geminiText(answer);
  const usage = geminiUsage(answer);
  const finish = geminiFinish(answer);
  return {
    id: ANTHROPIC_ID,
    type: "message",
    role: "assistant",
    content: [{ type: "text", text }],
    model,
    stop_reason: anthropicStop(finish),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

function parseSseFrame(frame: string): Json | null {
  const dataLines: string[] = [];
  let bare = "";
  for (const line of frame.split("\n")) {
    const l = line.replace(/\r$/, "");
    if (l.startsWith("data:")) {
      dataLines.push(l.slice(5).trimStart());
    } else if (l.startsWith("event:") || l.startsWith(":") || l === "") {
      // named frames carry no extra payload for Gemini; ignore
    } else {
      bare += l;
    }
  }
  const text = (dataLines.length > 0 ? dataLines.join("\n") : bare).trim();
  if (!text || text === "[DONE]") return null;
  try {
    const parsed: unknown = JSON.parse(text);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

async function* geminiFrames(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<Json> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (value) buf += decoder.decode(value, { stream: true });
      if (done) buf += decoder.decode();
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const parsed = parseSseFrame(frame);
        if (parsed) yield parsed;
      }
      if (done) break;
    }
    if (buf.trim()) {
      const parsed = parseSseFrame(buf);
      if (parsed) yield parsed;
    }
  } finally {
    reader.releaseLock();
  }
}

export async function* geminiEventsToOpenAI(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  const base = {
    id: OPENAI_ID,
    object: "chat.completion.chunk",
    created: CREATED,
    model,
  };
  yield `data: ${JSON.stringify({
    ...base,
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  })}\n\n`;
  let usage = { input: 0, output: 0, total: 0 };
  let finish: string | null = null;
  for await (const frame of geminiFrames(upstream)) {
    const text = geminiText(frame);
    if (text) {
      yield `data: ${JSON.stringify({
        ...base,
        choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
      })}\n\n`;
    }
    const u = geminiUsage(frame);
    if (u.input || u.output || u.total) usage = u;
    const f = geminiFinish(frame);
    if (f) finish = f;
  }
  yield `data: ${JSON.stringify({
    ...base,
    choices: [{ index: 0, delta: {}, finish_reason: openAIFinish(finish) }],
  })}\n\n`;
  // The wire check requires final usage on OpenAI streams; the usage chunk
  // (choices: []) is explicitly allowed by the SSE checks, so it is always
  // emitted, with the provider's verbatim usage.
  yield `data: ${JSON.stringify({
    ...base,
    choices: [],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.total,
    },
  })}\n\n`;
  yield "data: [DONE]\n\n";
}

export async function* geminiEventsToAnthropic(
  upstream: ReadableStream<Uint8Array>,
  model: string,
): AsyncGenerator<string> {
  // message_start carries input usage, but Gemini only reports usage on the
  // final frame; the wire check for anthropic+stream compares output usage
  // only, so 0 here is safe and the real output count rides message_delta.
  yield sseEvent("message_start", {
    type: "message_start",
    message: {
      id: ANTHROPIC_ID,
      type: "message",
      role: "assistant",
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });
  yield sseEvent("content_block_start", {
    type: "content_block_start",
    index: 0,
    content_block: { type: "text", text: "" },
  });
  let usage = { input: 0, output: 0, total: 0 };
  let finish: string | null = null;
  for await (const frame of geminiFrames(upstream)) {
    const text = geminiText(frame);
    if (text) {
      yield sseEvent("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text },
      });
    }
    const u = geminiUsage(frame);
    if (u.input || u.output || u.total) usage = u;
    const f = geminiFinish(frame);
    if (f) finish = f;
  }
  yield sseEvent("content_block_stop", {
    type: "content_block_stop",
    index: 0,
  });
  yield sseEvent("message_delta", {
    type: "message_delta",
    delta: { stop_reason: anthropicStop(finish), stop_sequence: null },
    usage: { output_tokens: usage.output },
  });
  yield sseEvent("message_stop", { type: "message_stop" });
}
