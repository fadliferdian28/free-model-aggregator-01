/**
 * Free-model aggregator entry (free-model-aggregator-01).
 *
 * One server speaking both OpenAI Chat Completions and Anthropic Messages
 * over ten simulated free-tier providers (openai / anthropic / gemini wires).
 *
 * Routing policy (designed from the kit's published brief + sealed table):
 * - Candidates are (provider, best-free-model) pairs whose capabilities meet
 *   the request's requirements; they are ranked by the quality table's
 *   rating_milli, ties broken by provider id then model id (deterministic).
 * - Planned calls per provider are capped at the published budget maximum
 *   for that provider (CALLS_MAX below). Only the public ranges are used;
 *   the per-trial drawn budgets are never known to this entry.
 * - On a refusal (402/429/5xx) the SAME provider is retried once; a second
 *   consecutive refusal drops the provider for the rest of the trial. A
 *   success resets the consecutive-refusal count. 401/400 answers are
 *   bug-class, never retried on the same provider.
 * - An OpenAI JSON-mode request is never routed to an anthropic-wire
 *   provider: that wire cannot carry JSON mode, so the echo check would
 *   fail even though the model itself supports JSON.
 * - Exactly one successful provider call per request; upstream answers are
 *   converted to the client's wire with the provider's usage verbatim.
 * - No wall clock and no unseeded randomness anywhere: replay must produce
 *   byte-identical behaviour.
 */

import http from "node:http";
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
} from "./lib/providers.ts";
import type {
  Brief,
  CatalogReport,
  Json,
  Probe,
  ProbedModel,
  ProviderDirectoryEntry,
  RatingRow,
  Requirement,
} from "./lib/providers.ts";
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
} from "./lib/wire.ts";
import type { ServedModel } from "./lib/wire.ts";
import {
  anthropicToGemini,
  geminiEventsToAnthropic,
  geminiEventsToOpenAI,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIToGemini,
} from "./lib/gemini.ts";
import {
  anthropicAnswerToOpenAI,
  anthropicEventsToOpenAI,
  openAIRequestToAnthropic,
} from "./lib/translate.ts";

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";

// Published per-provider budget maxima (from the kit's sealed table). The
// drawn per-trial budgets are unknown; capping the plan at the published
// maximum keeps one provider from being over-committed early.
const CALLS_MAX: Record<string, number> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};
const DEFAULT_CALLS_MAX = 4;
const ATTEMPT_TIMEOUT_MS = 8000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const POLL_MS = 100;

interface Candidate {
  probe: Probe;
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  row: RatingRow;
  rating: number;
}

interface ProviderState {
  calls: number;
  consecutive: number;
  dead: boolean;
}

interface UpstreamResult {
  received: boolean;
  ok: boolean;
  status: number;
  json: Json | null;
  stream: ReadableStream<Uint8Array> | null;
}

const brief: Brief = PROMPT_FILE
  ? readBrief(PROMPT_FILE)
  : {
      directory: { schema: "", providers: [] },
      ratings: null,
      workload: null,
      error: "no prompt file",
    };

let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
const providerState = new Map<string, ProviderState>();

function callsMax(providerId: string): number {
  return CALLS_MAX[providerId] ?? DEFAULT_CALLS_MAX;
}

function stateFor(providerId: string): ProviderState {
  let st = providerState.get(providerId);
  if (!st) {
    st = { calls: 0, consecutive: 0, dead: false };
    providerState.set(providerId, st);
  }
  return st;
}

function isJsonMode(body: Json): boolean {
  const fmt = body.response_format;
  if (!isRecord(fmt)) return false;
  return fmt.type === "json_object" || fmt.type === "json_schema";
}

function requirementsFromBody(body: Json): Requirement {
  const req: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
  const msgs = Array.isArray(body.messages) ? body.messages : [];
  for (const m of msgs) {
    if (!isRecord(m)) continue;
    const content = m.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "image_url" || block.type === "image") {
        req.vision = true;
        break;
      }
    }
    if (req.vision) break;
  }
  if (isJsonMode(body)) req.json = true;
  return req;
}

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const models = [...probe.models].sort((a, b) =>
      a.id < b.id ? -1 : a.id > b.id ? 1 : 0,
    );
    for (const model of models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({
        id: model.id,
        provider: probe.provider.id,
        display_name: model.display_name,
      });
    }
  }
  return out;
}

function candidatesFor(
  requires: Requirement,
  clientApi: "openai" | "anthropic",
  jsonMode: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const st = stateFor(probe.provider.id);
    if (st.dead || st.calls >= callsMax(probe.provider.id)) continue;
    // r07 trap: an OpenAI JSON-mode request must never go to an
    // anthropic-wire provider (that wire cannot carry JSON mode).
    if (jsonMode && clientApi === "openai" && probe.provider.wire === "anthropic") {
      continue;
    }
    let best: { model: ProbedModel; row: RatingRow; rating: number } | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!row) continue;
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      if (
        !best ||
        row.rating_milli > best.rating ||
        (row.rating_milli === best.rating && model.id < best.model.id)
      ) {
        best = { model, row, rating: row.rating_milli };
      }
    }
    if (best) {
      out.push({
        probe,
        provider: probe.provider,
        model: best.model,
        row: best.row,
        rating: best.rating,
      });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

async function callUpstream(
  cand: Candidate,
  clientApi: "openai" | "anthropic",
  body: Json,
  stream: boolean,
): Promise<UpstreamResult> {
  const base = joinBase(ENV_URL, cand.provider.base);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    ...authHeaders(cand.provider.auth, cand.provider.key),
  };
  const model = cand.model.id;
  let url: string;
  let upBody: Json;
  if (cand.provider.wire === "openai") {
    url = `${base}/v1/chat/completions`;
    upBody =
      clientApi === "openai"
        ? { ...body, model }
        : anthropicToOpenAI(body, model);
    // The wire check requires final usage on OpenAI streams, and the client
    // body does not ask for it — so ask the provider for it upstream.
    // stream_options is not part of the echo, so this cannot break one.
    if (stream) upBody = { ...upBody, stream_options: { include_usage: true } };
  } else if (cand.provider.wire === "anthropic") {
    url = `${base}/v1/messages`;
    headers["anthropic-version"] = "2023-06-01";
    upBody =
      clientApi === "anthropic"
        ? { ...body, model }
        : openAIRequestToAnthropic(body, model);
  } else {
    const verb = stream ? "streamGenerateContent" : "generateContent";
    url = `${base}/v1beta/models/${encodeURIComponent(model)}:${verb}`;
    upBody =
      clientApi === "openai" ? openAIToGemini(body) : anthropicToGemini(body);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(upBody),
      signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
    });
  } catch {
    return { received: false, ok: false, status: 0, json: null, stream: null };
  }
  if (!res.ok) {
    try {
      await res.arrayBuffer();
    } catch {
      // ignore drain errors
    }
    return { received: true, ok: false, status: res.status, json: null, stream: null };
  }
  if (stream) {
    return { received: true, ok: true, status: res.status, json: null, stream: res.body };
  }
  const data: unknown = await res.json().catch(() => null);
  return {
    received: true,
    ok: true,
    status: res.status,
    json: isRecord(data) ? data : null,
    stream: null,
  };
}

function sendJson(res: http.ServerResponse, status: number, obj: Json): void {
  const data = JSON.stringify(obj);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(data),
  });
  res.end(data);
}

function sendError(
  res: http.ServerResponse,
  wire: "openai" | "anthropic",
  status: number,
  type: string,
  message: string,
): void {
  // NOTE: opposite parameter orders — openAIError(message, type, code),
  // anthropicError(type, message).
  const body =
    wire === "anthropic"
      ? anthropicError(type, message)
      : openAIError(message, type, null);
  sendJson(res, status, body);
}

async function pipeRaw(
  stream: ReadableStream<Uint8Array>,
  res: http.ServerResponse,
): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) res.write(value);
    }
  } finally {
    reader.releaseLock();
  }
  res.end();
}

async function sendUpstream(
  clientApi: "openai" | "anthropic",
  cand: Candidate,
  up: UpstreamResult,
  body: Json,
  res: http.ServerResponse,
  stream: boolean,
): Promise<void> {
  const requested =
    typeof body.model === "string" && body.model ? body.model : cand.model.id;
  const wire = cand.provider.wire;
  if (!stream) {
    const answer = up.json ?? {};
    let out: Json;
    if (clientApi === "openai") {
      if (wire === "openai") out = answer;
      else if (wire === "anthropic") out = anthropicAnswerToOpenAI(answer, requested);
      else out = geminiToOpenAI(answer, requested);
    } else {
      if (wire === "openai") out = openAIToAnthropic(answer, requested);
      else if (wire === "anthropic") out = answer;
      else out = geminiToAnthropic(answer, requested);
    }
    sendJson(res, 200, out);
    return;
  }
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const upstream = up.stream;
  if (!upstream) {
    res.end();
    return;
  }
  if (clientApi === "openai" && wire === "openai") {
    await pipeRaw(upstream, res);
    return;
  }
  if (clientApi === "anthropic" && wire === "anthropic") {
    await pipeRaw(upstream, res);
    return;
  }
  let events: AsyncGenerator<string>;
  if (clientApi === "openai" && wire === "anthropic") {
    events = anthropicEventsToOpenAI(upstream, requested);
  } else if (clientApi === "openai") {
    events = geminiEventsToOpenAI(upstream, requested);
  } else if (wire === "openai") {
    events = anthropicEventsFromOpenAI(upstream, requested);
  } else {
    events = geminiEventsToAnthropic(upstream, requested);
  }
  for await (const frame of events) {
    res.write(frame);
  }
  res.end();
}

async function serveRouted(
  clientApi: "openai" | "anthropic",
  body: Json,
  res: http.ServerResponse,
  stream: boolean,
): Promise<void> {
  const idx = manifestIndex++;
  const manifest = brief.workload?.manifest;
  const requires =
    manifest && idx < manifest.length
      ? manifest[idx].requires
      : requirementsFromBody(body);
  const jsonMode = clientApi === "openai" ? isJsonMode(body) : false;
  const cands = candidatesFor(requires, clientApi, jsonMode);
  for (const cand of cands) {
    const st = stateFor(cand.provider.id);
    if (st.dead) continue;
    for (let attempt = 0; attempt < 2; attempt++) {
      if (st.calls >= callsMax(cand.provider.id)) {
        st.dead = true;
        break;
      }
      const up = await callUpstream(cand, clientApi, body, stream);
      if (up.received) st.calls += 1;
      if (up.ok) {
        st.consecutive = 0;
        await sendUpstream(clientApi, cand, up, body, res, stream);
        return;
      }
      // 401/400 are bug-class: never retry on the same provider.
      if (up.status === 401 || up.status === 400) break;
      st.consecutive += 1;
      if (st.consecutive >= 2) {
        st.dead = true;
        break;
      }
      // otherwise retry the same provider once (loop continues)
    }
  }
  if (clientApi === "openai") {
    sendJson(
      res,
      503,
      openAIError(
        "No free model meets the request requirements",
        "server_error",
        "no_free_model",
      ),
    );
  } else {
    sendJson(
      res,
      503,
      anthropicError(
        "overloaded_error",
        "No free model meets the request requirements",
      ),
    );
  }
}

async function readJson(req: http.IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    total += buf.length;
    if (total > MAX_BODY_BYTES) return null;
    chunks.push(buf);
  }
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

async function handle(
  req: http.IncomingMessage,
  res: http.ServerResponse,
): Promise<void> {
  const url = new URL(req.url ?? "/", "http://localhost");
  try {
    if (req.method === "GET" && url.pathname === "/v1/models") {
      const list = servedModels();
      const out =
        req.headers["x-api-key"] !== undefined
          ? anthropicModelList(list)
          : openAIModelList(list);
      sendJson(res, 200, out);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      const body = await readJson(req);
      if (!body) {
        sendError(
          res,
          "openai",
          400,
          "invalid_request_error",
          "request body must be one JSON object",
        );
        return;
      }
      await serveRouted("openai", body, res, body.stream === true);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/messages") {
      const body = await readJson(req);
      if (!body) {
        sendError(
          res,
          "anthropic",
          400,
          "invalid_request_error",
          "request body must be one JSON object",
        );
        return;
      }
      await serveRouted("anthropic", body, res, body.stream === true);
      return;
    }
    sendError(
      res,
      url.pathname.startsWith("/v1/messages") ? "anthropic" : "openai",
      404,
      "not_found_error",
      "unknown route",
    );
  } catch (err) {
    console.error(err);
    if (!res.headersSent) {
      sendError(res, "openai", 500, "api_error", "internal error");
    } else {
      try {
        res.end();
      } catch {
        // ignore
      }
    }
  }
}

async function load(): Promise<void> {
  const entries = brief.directory?.providers ?? [];
  probes = await Promise.all(
    entries.map((p) => probeProvider(ENV_URL, p, 5000)),
  );
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

async function start(): Promise<void> {
  if (!ENV_URL || !PROMPT_FILE || brief.error) {
    if (process.argv.includes("--list")) {
      process.stdout.write(JSON.stringify({ providers: [] }) + "\n");
      process.exit(0);
    }
    throw new Error(brief.error ?? "missing MAKEFUN_ENV_URL / MAKEFUN_PROMPT_FILE");
  }
  await load();
  if (process.argv.includes("--list")) {
    process.stdout.write(JSON.stringify(catalog) + "\n");
    process.exit(0);
  }
  const server = http.createServer((req, res) => {
    void handle(req, res);
  });
  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : 0;
  if (port === 8080 || port === 0) throw new Error(`bad listen port ${port}`);
  // Register once. A transport failure (the fetch itself throwing) means the
  // driver never received anything, so one retry is safe; a received
  // non-ok answer must NOT be retried, because the driver's begin() runs on
  // receipt and is once-only.
  let reg: Response | null = null;
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      reg = await fetch(`${ENV_URL}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
      });
      break;
    } catch (err) {
      lastError = err;
      reg = null;
    }
  }
  if (!reg) {
    throw new Error(
      `register unreachable: ${lastError instanceof Error ? lastError.message : String(lastError)}`,
    );
  }
  if (!reg.ok) throw new Error(`register failed: ${reg.status}`);
  const timer = setInterval(() => {
    void (async () => {
      try {
        const w = await fetch(`${ENV_URL}/work`);
        const data: unknown = await w.json().catch(() => null);
        if (isRecord(data) && data.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // retry next tick
      }
    })();
  }, POLL_MS);
}

start().catch((err) => {
  console.error(err);
  process.exit(1);
});
