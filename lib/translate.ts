/**
 * OpenAI <-> Anthropic translation for the missing directions
 * (lib/translate.ts).
 *
 * The starter's wire.ts covers anthropic-request -> openai-request and
 * openai-answer -> anthropic-answer. This file covers the other two
 * directions, needed to serve OpenAI clients from anthropic-wire providers
 * (zai) and to convert anthropic-wire answers back to OpenAI clients:
 *
 * - openAIRequestToAnthropic: Chat Completions request -> Messages request.
 *   System messages join with \n into `system`; images become Anthropic
 *   image blocks (base64 source for data URLs, url source otherwise);
 *   assistant tool_calls become tool_use blocks; tool messages become
 *   tool_result blocks in a user message. OpenAI JSON mode has NO Anthropic
 *   wire equivalent, so response_format is dropped — the router never sends
 *   an OpenAI JSON request to an anthropic-wire provider, so this path
 *   should never carry JSON mode in the first place.
 * - anthropicAnswerToOpenAI: Messages JSON answer -> Chat Completions JSON
 *   answer (text concatenated, usage mapped input/output -> prompt/
 *   completion tokens verbatim, stop_reason mapped to finish_reason).
 * - anthropicEventsToOpenAI: Anthropic SSE -> OpenAI SSE event strings
 *   (role chunk, content deltas, one finish chunk, optional usage chunk,
 *   [DONE] last and once).
 *
 * Fixed ids and the fixed created timestamp keep replay deterministic.
 */

import { isRecord } from "./providers.ts";
import type { Json } from "./providers.ts";

const CREATED = 1767225600;
const OPENAI_ID = "chatcmpl-anthropic-1";

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

function openAIImageBlock(content: unknown): Json | null {
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (!isRecord(block) || block.type !== "image_url") continue;
    const iu = isRecord(block.image_url) ? block.image_url : null;
    const url = iu && typeof iu.url === "string" ? iu.url : "";
    if (!url) continue;
    const m = /^data:([^;]+);base64,(.*)$/.exec(url);
    if (m) {
      return {
        type: "image",
        source: { type: "base64", media_type: m[1], data: m[2] },
      };
    }
    return { type: "image", source: { type: "url", url } };
  }
  return null;
}

function openAIStops(body: Json): string[] {
  const s = body.stop;
  if (typeof s === "string") return s ? [s] : [];
  if (Array.isArray(s)) {
    return s.filter((x): x is string => typeof x === "string");
  }
  return [];
}

function anthropicToolChoice(tc: unknown): Json | null {
  if (tc === "auto") return { type: "auto" };
  if (tc === "none") return { type: "none" };
  if (
    isRecord(tc) &&
    tc.type === "function" &&
    isRecord(tc.function) &&
    typeof tc.function.name === "string"
  ) {
    return { type: "tool", name: tc.function.name };
  }
  return null;
}

function mergeAnthropicMessages(msgs: Json[]): Json[] {
  const out: Json[] = [];
  for (const m of msgs) {
    const last = out[out.length - 1];
    if (last && last.role === m.role) {
      const lc = Array.isArray(last.content) ? last.content : [];
      const mc = Array.isArray(m.content) ? m.content : [];
      last.content = [...lc, ...mc];
    } else {
      out.push({
        role: m.role,
        content: Array.isArray(m.content) ? [...m.content] : m.content,
      });
    }
  }
  return out;
}

export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  for (const m of msgs) {
    if (!isRecord(m)) continue;
    const role = m.role;
    if (role === "system") {
      const t = openAIText(m.content);
      if (t) systemParts.push(t);
      continue;
    }
    if (role === "tool") {
      const content =
        typeof m.content === "string"
          ? m.content
          : JSON.stringify(m.content ?? "");
      messages.push({
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id:
              typeof m.tool_call_id === "string" ? m.tool_call_id : "tool",
            content,
          },
        ],
      });
      continue;
    }
    const blocks: Json[] = [];
    const text = openAIText(m.content);
    if (text) blocks.push({ type: "text", text });
    const image = openAIImageBlock(m.content);
    if (image) blocks.push(image);
    if (role === "assistant" && Array.isArray(m.tool_calls)) {
      for (const tc of m.tool_calls) {
        if (!isRecord(tc)) continue;
        const fn = isRecord(tc.function) ? tc.function : {};
        const name = typeof fn.name === "string" ? fn.name : "";
        if (!name) continue;
        let input: Json = {};
        if (typeof fn.arguments === "string") {
          try {
            const parsed: unknown = JSON.parse(fn.arguments);
            if (isRecord(parsed)) input = parsed;
          } catch {
            // keep empty input
          }
        } else if (isRecord(fn.arguments)) {
          input = fn.arguments;
        }
        blocks.push({
          type: "tool_use",
          id: typeof tc.id === "string" ? tc.id : name,
          name,
          input,
        });
      }
    }
    if (blocks.length === 0) continue;
    messages.push({
      role: role === "assistant" ? "assistant" : "user",
      content: blocks,
    });
  }
  const out: Json = { model, messages: mergeAnthropicMessages(messages) };
  const system = systemParts.length ? systemParts.join("\n") : null;
  if (system) out.system = system;
  const tools = Array.isArray(body.tools) ? body.tools : [];
  const anthropicTools: Json[] = [];
  for (const t of tools) {
    if (!isRecord(t) || t.type !== "function") continue;
    const fn = isRecord(t.function) ? t.function : {};
    const name = typeof fn.name === "string" ? fn.name : "";
    if (!name) continue;
    anthropicTools.push({
      name,
      description: typeof fn.description === "string" ? fn.description : "",
      input_schema: isRecord(fn.parameters) ? fn.parameters : { type: "object" },
    });
  }
  if (anthropicTools.length) out.tools = anthropicTools;
  const maxTokens = body.max_tokens ?? body.max_completion_tokens;
  if (typeof maxTokens === "number") out.max_tokens = maxTokens;
  const stops = openAIStops(body);
  if (stops.length) out.stop_sequences = stops;
  if (body.stream === true) out.stream = true;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  const choice = anthropicToolChoice(body.tool_choice);
  if (choice) out.tool_choice = choice;
  return out;
}

function openAIFinishFromAnthropic(reason: string | null): string {
  switch (reason) {
    case "max_tokens":
      return "length";
    case "tool_use":
      return "tool_calls";
    case "refusal":
      return "content_filter";
    default:
      return "stop";
  }
}

export function anthropicAnswerToOpenAI(answer: Json, model: string): Json {
  let text = "";
  const content = Array.isArray(answer.content) ? answer.content : [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      text += block.text;
    }
  }
  const usage = isRecord(answer.usage) ? answer.usage : {};
  const input = typeof usage.input_tokens === "number" ? usage.input_tokens : 0;
  const output =
    typeof usage.output_tokens === "number" ? usage.output_tokens : 0;
  const stop = typeof answer.stop_reason === "string" ? answer.stop_reason : null;
  return {
    id: OPENAI_ID,
    object: "chat.completion",
    created: CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: openAIFinishFromAnthropic(stop),
      },
    ],
    usage: {
      prompt_tokens: input,
      completion_tokens: output,
      total_tokens: input + output,
    },
  };
}

interface SseFrame {
  event: string | null;
  data: Json | null;
}

function parseNamedFrame(frame: string): SseFrame {
  let event: string | null = null;
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    const l = line.replace(/\r$/, "");
    if (l.startsWith("event:")) {
      event = l.slice(6).trim();
    } else if (l.startsWith("data:")) {
      dataLines.push(l.slice(5).trimStart());
    }
  }
  const text = dataLines.join("\n").trim();
  if (!text || text === "[DONE]") return { event, data: null };
  try {
    const parsed: unknown = JSON.parse(text);
    return { event, data: isRecord(parsed) ? parsed : null };
  } catch {
    return { event, data: null };
  }
}

async function* anthropicFrames(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<SseFrame> {
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
        yield parseNamedFrame(frame);
      }
      if (done) break;
    }
    if (buf.trim()) {
      yield parseNamedFrame(buf);
    }
  } finally {
    reader.releaseLock();
  }
}

export async function* anthropicEventsToOpenAI(
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
  let input = 0;
  let output = 0;
  let stop: string | null = null;
  for await (const frame of anthropicFrames(upstream)) {
    const data = frame.data;
    if (!data) continue;
    const type = typeof data.type === "string" ? data.type : "";
    if (type === "message_start") {
      const msg = isRecord(data.message) ? data.message : {};
      const u = isRecord(msg.usage) ? msg.usage : {};
      if (typeof u.input_tokens === "number") input = u.input_tokens;
    } else if (type === "content_block_delta") {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (
        delta.type === "text_delta" &&
        typeof delta.text === "string" &&
        delta.text
      ) {
        yield `data: ${JSON.stringify({
          ...base,
          choices: [
            { index: 0, delta: { content: delta.text }, finish_reason: null },
          ],
        })}\n\n`;
      }
    } else if (type === "message_delta") {
      const delta = isRecord(data.delta) ? data.delta : {};
      if (typeof delta.stop_reason === "string") stop = delta.stop_reason;
      const u = isRecord(data.usage) ? data.usage : {};
      if (typeof u.output_tokens === "number") output = u.output_tokens;
    }
  }
  yield `data: ${JSON.stringify({
    ...base,
    choices: [
      {
        index: 0,
        delta: {},
        finish_reason: openAIFinishFromAnthropic(stop),
      },
    ],
  })}\n\n`;
  // The wire check requires final usage on OpenAI streams; the usage chunk
  // (choices: []) is explicitly allowed by the SSE checks, so it is always
  // emitted, with the provider's verbatim usage.
  yield `data: ${JSON.stringify({
    ...base,
    choices: [],
    usage: {
      prompt_tokens: input,
      completion_tokens: output,
      total_tokens: input + output,
    },
  })}\n\n`;
  yield "data: [DONE]\n\n";
}
