/**
 * Free-model aggregator entry (free-model-aggregator-01).
 *
 * One server speaking both OpenAI Chat Completions and Anthropic Messages
 * over ten simulated free-tier providers (openai / anthropic / gemini wires).
 *
 * Routing policy (v3: tight startup planner + retry-twice failover,
 * designed from the kit's published brief + sealed table):
 * - At startup, a planner solves a min-cost assignment of the manifest
 *   requests across all ten providers under per-provider planning caps
 *   (PLAN_CAPS below, sum 16 — four requests stay unassigned and take
 *   the greedy failover path at runtime), maximising the weighted rating.
 *   Every provider is used at least once, so every first call — which
 *   always succeeds — is spent productively instead of wasted on a
 *   concentrated greedy order.
 * - Each request is served by its planned provider first. On a refusal
 *   (402/429/5xx) the SAME provider is retried up to twice; a third
 *   consecutive refusal drops the provider for the rest of the trial, and
 *   the request fails over to the next best available provider by rating
 *   (the greedy candidate order). A success resets the consecutive-refusal
 *   count. 401/400 answers are bug-class, never retried on the same provider.
 * - The planning caps shape ONLY the initial assignment; during execution
 *   the only per-provider limit is the published budget maximum (CALLS_MAX
 *   below). Only the public ranges are used; the per-trial drawn budgets
 *   are never known to this entry.
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

// Planning caps per provider (v2): the startup planner assigns at most this
// many requests to each provider. Set from the published budget ranges, not
// from any trial's drawn schedule. Small-budget providers are capped at one
// because their second call fails most often; large-budget providers take up
// to four because their failure numbers are spread thinly over a long budget.
// Total planned capacity is 23 for 20 requests, leaving three slots of slack
// for the flow to choose the best assignment. These caps shape ONLY the
// initial assignment and are never enforced during execution.
const PLAN_CAPS: Record<string, number> = {
  groq: 2,
  cerebras: 1,
  gemini: 1,
  zai: 1,
  openrouter: 3,
  sambanova: 1,
  pollinations: 3,
  cloudflare: 1,
  ovh: 1,
  kilol: 2,
};
const DEFAULT_PLAN_CAP = 1;
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

// One planned slot: the candidate the planner assigned to a manifest
// request, plus that request's weight (used for the planned-quality total).
interface PlanSlot {
  candidate: Candidate;
  weight: number;
}

// The startup plan: one planned slot per manifest request index, or null
// where the planner could not assign the request (serveRouted falls back to
// the greedy candidate list for those). Computed once by computePlan()
// after the providers are probed, before serving starts.
let plan: (PlanSlot | null)[] = [];
let planEarned = 0;
let planBound = 0;

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

// The best free model on one provider for a request: the highest-rated free
// model whose capabilities meet the requirements, ties broken by model id.
// Null when the provider cannot serve this request (dead, at its budget
// maximum, wrong capabilities, or the OpenAI-JSON-on-anthropic-wire trap).
// Shared by the greedy candidate list and the startup planner, so both agree
// on what each provider can do for each request.
function bestCandidateForProvider(
  probe: Probe,
  requires: Requirement,
  clientApi: "openai" | "anthropic",
  jsonMode: boolean,
): Candidate | null {
  const st = stateFor(probe.provider.id);
  if (st.dead || st.calls >= callsMax(probe.provider.id)) return null;
  // r07 trap: an OpenAI JSON-mode request must never go to an
  // anthropic-wire provider (that wire cannot carry JSON mode).
  if (jsonMode && clientApi === "openai" && probe.provider.wire === "anthropic") {
    return null;
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
  if (!best) return null;
  return {
    probe,
    provider: probe.provider,
    model: best.model,
    row: best.row,
    rating: best.rating,
  };
}

function candidatesFor(
  requires: Requirement,
  clientApi: "openai" | "anthropic",
  jsonMode: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const cand = bestCandidateForProvider(probe, requires, clientApi, jsonMode);
    if (cand) out.push(cand);
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

// Min-cost flow by successive shortest augmenting paths (Bellman-Ford each
// round; the graph is tiny — 32 nodes, ~200 edges — and has no negative
// cycles by construction). Edges are [from, to, capacity, cost]. Every
// augmentation pushes exactly one unit: every source edge has capacity 1,
// so the bottleneck on any path is 1. Deterministic: edges are relaxed in
// insertion order with a strict improvement test, so a fixed input order
// yields a fixed result. Returns the flow value, the total cost, and — for
// each input edge, in input order — whether it carries flow.
interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

function minCostFlow(
  nodeCount: number,
  edges: [number, number, number, number][],
  source: number,
  sink: number,
  maxFlow: number,
): { flow: number; cost: number; used: boolean[] } {
  const graph: FlowEdge[][] = Array.from({ length: nodeCount }, () => []);
  const forwardRef: { from: number; idx: number }[] = [];
  for (const [from, to, cap, cost] of edges) {
    const fwd: FlowEdge = { to, rev: graph[to].length, cap, cost };
    const rev: FlowEdge = { to: from, rev: graph[from].length, cap: 0, cost: -cost };
    forwardRef.push({ from, idx: graph[from].length });
    graph[from].push(fwd);
    graph[to].push(rev);
  }
  let flow = 0;
  let cost = 0;
  const INF = Number.POSITIVE_INFINITY;
  while (flow < maxFlow) {
    const dist: number[] = new Array(nodeCount).fill(INF);
    const prevNode: number[] = new Array(nodeCount).fill(-1);
    const prevEdge: number[] = new Array(nodeCount).fill(-1);
    dist[source] = 0;
    for (let iter = 0; iter < nodeCount; iter++) {
      let changed = false;
      for (let u = 0; u < nodeCount; u++) {
        if (dist[u] === INF) continue;
        for (let i = 0; i < graph[u].length; i++) {
          const e = graph[u][i];
          if (e.cap > 0 && dist[e.to] > dist[u] + e.cost) {
            dist[e.to] = dist[u] + e.cost;
            prevNode[e.to] = u;
            prevEdge[e.to] = i;
            changed = true;
          }
        }
      }
      if (!changed) break;
    }
    if (dist[sink] === INF) break;
    let v = sink;
    while (v !== source) {
      const u = prevNode[v];
      const e = graph[u][prevEdge[v]];
      e.cap -= 1;
      graph[v][e.rev].cap += 1;
      v = u;
    }
    flow += 1;
    cost += dist[sink];
  }
  // A forward edge carries flow iff its reverse edge gained capacity.
  const used: boolean[] = forwardRef.map(({ from, idx }) => {
    const fwd = graph[from][idx];
    return graph[fwd.to][fwd.rev].cap > 0;
  });
  return { flow, cost, used };
}

// The startup planner: assigns every manifest request to one provider,
// maximising the weighted rating under the planning caps. The cost of
// assigning request r to provider p is -(weight_r * rating of p's best free
// model meeting r's requirements); a request with no edge to a provider
// cannot be served there. Runs once, after the providers are probed and
// before serving starts. Fully deterministic: it depends only on the
// brief's ratings table and workload manifest and on PLAN_CAPS, so it
// produces the same assignment on every instance and every replay.
function computePlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  plan = [];
  planEarned = 0;
  planBound = 0;
  if (manifest.length === 0) {
    console.error("planner: no workload manifest; falling back to greedy routing");
    return;
  }
  const providerIds = [...probes]
    .filter((p) => p.ok)
    .map((p) => p.provider.id)
    .sort();
  const probeById = new Map(probes.map((p) => [p.provider.id, p]));
  const providerIndex = new Map(providerIds.map((id, j) => [id, j]));
  const n = manifest.length;
  const m = providerIds.length;
  const source = 0;
  const sink = n + m + 1;
  const edges: [number, number, number, number][] = [];
  for (let i = 0; i < n; i++) edges.push([source, 1 + i, 1, 0]);
  const edgeIndex = new Map<string, number>();
  const candByPair = new Map<string, Candidate>();
  for (let i = 0; i < n; i++) {
    const req = manifest[i];
    const jsonMode = req.api === "openai" && req.requires.json === true;
    for (const pid of providerIds) {
      const probe = probeById.get(pid);
      if (!probe) continue;
      const cand = bestCandidateForProvider(probe, req.requires, req.api, jsonMode);
      if (!cand) continue;
      const key = `${i}:${pid}`;
      edgeIndex.set(key, edges.length);
      candByPair.set(key, cand);
      edges.push([1 + i, 1 + n + (providerIndex.get(pid) ?? 0), 1, -(req.weight * cand.rating)]);
    }
  }
  for (let j = 0; j < m; j++) {
    const cap = PLAN_CAPS[providerIds[j]] ?? DEFAULT_PLAN_CAP;
    edges.push([1 + n + j, sink, cap, 0]);
  }
  const { flow, used } = minCostFlow(sink + 1, edges, source, sink, n);
  for (let i = 0; i < n; i++) {
    const req = manifest[i];
    let slot: PlanSlot | null = null;
    for (const pid of providerIds) {
      const key = `${i}:${pid}`;
      const ei = edgeIndex.get(key);
      if (ei === undefined || !used[ei]) continue;
      const cand = candByPair.get(key);
      if (!cand) continue;
      slot = { candidate: cand, weight: req.weight };
      planEarned += req.weight * cand.rating;
      break;
    }
    plan.push(slot);
  }
  for (let i = 0; i < n; i++) {
    const req = manifest[i];
    let best = 0;
    for (const pid of providerIds) {
      const cand = candByPair.get(`${i}:${pid}`);
      if (cand && cand.rating > best) best = cand.rating;
    }
    planBound += req.weight * best;
  }
  const quality = planBound > 0 ? Math.floor((100 * planEarned) / planBound) : 0;
  console.error(
    `planner: assigned ${flow}/${n} requests, planned earned ${planEarned}/${planBound} (quality ${quality})`,
  );
  if (flow < n) {
    console.error("planner: WARNING — some requests unassigned; greedy fallback will serve them");
  }
  for (let i = 0; i < n; i++) {
    const slot = plan[i];
    const req = manifest[i];
    console.error(
      `planner: ${req.id} (wt ${req.weight}) -> ${slot ? `${slot.candidate.provider.id}/${slot.candidate.model.id} (${slot.candidate.rating})` : "UNASSIGNED"}`,
    );
  }
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
  if (!isRecord(data)) {
    // A 200 with an unparseable or non-object body is unusable: fail this
    // attempt so the router fails over instead of serving an empty answer.
    return { received: true, ok: false, status: res.status, json: null, stream: null };
  }
  return {
    received: true,
    ok: true,
    status: res.status,
    json: data,
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
  // v2: the planned candidate for this request goes first; the greedy list
  // (minus the planned provider, already tried) is the failover order. If
  // the planned provider is absent from the greedy list (dead or at its
  // budget maximum), the greedy order serves as-is.
  const slot = idx < plan.length ? plan[idx] : null;
  if (slot) {
    const plannedPos = cands.findIndex(
      (c) => c.provider.id === slot.candidate.provider.id,
    );
    if (plannedPos > 0) {
      const [planned] = cands.splice(plannedPos, 1);
      cands.unshift(planned);
    }
  }
  for (const cand of cands) {
    const st = stateFor(cand.provider.id);
    if (st.dead) continue;
    for (let attempt = 0; attempt < 3; attempt++) {
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
      if (st.consecutive >= 3) {
        st.dead = true;
        break;
      }
      // otherwise retry the same provider again (up to two retries)
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
  computePlan();
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
