#!/usr/bin/env node
import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";

const MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market";

function parseArgs(argv) {
  const args = {
    config: "config/paper_btc_5m.json",
    outputDir: "reports/live_data_probes",
    durationSeconds: 60,
    assets: [],
    selfTest: false,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === "--config") {
      args.config = value;
      index += 1;
    } else if (key === "--output-dir") {
      args.outputDir = value;
      index += 1;
    } else if (key === "--duration-seconds") {
      args.durationSeconds = Number(value);
      index += 1;
    } else if (key === "--asset-id") {
      args.assets.push(String(value));
      index += 1;
    } else if (key === "--self-test") {
      args.selfTest = true;
    } else if (key === "--help" || key === "-h") {
      console.log(
        "Usage: node scripts/probe_polymarket_websocket.mjs [--config config/paper_btc_5m.json] [--duration-seconds 60] [--asset-id TOKEN_ID] [--self-test]",
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${key}`);
    }
  }
  if (!Number.isFinite(args.durationSeconds) || args.durationSeconds <= 0) {
    throw new Error("--duration-seconds must be positive");
  }
  return args;
}

function utcStamp(date = new Date()) {
  return date.toISOString().replaceAll("-", "").replaceAll(":", "").replace(/\.\d{3}Z$/, "Z");
}

async function loadJson(filePath) {
  return JSON.parse(await fsp.readFile(filePath, "utf8"));
}

function intervalStart(nowSeconds, intervalSeconds) {
  return Math.floor(nowSeconds / intervalSeconds) * intervalSeconds;
}

function parseJsonishList(value) {
  if (Array.isArray(value)) {
    return value;
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
}

function selectBinaryMarket(event) {
  const markets = Array.isArray(event.markets) ? event.markets : [];
  for (const market of markets) {
    const outcomes = parseJsonishList(market.outcomes).map((item) => String(item).toLowerCase());
    if (outcomes.includes("up") && outcomes.includes("down")) {
      return market;
    }
  }
  if (markets.length > 0) {
    return markets[0];
  }
  return event.clobTokenIds || event.clob_token_ids ? event : null;
}

function marketFromEvent(event, startEpoch, intervalSeconds) {
  const market = selectBinaryMarket(event);
  if (!market) {
    return null;
  }
  const outcomes = parseJsonishList(market.outcomes).map((item) => String(item).toLowerCase());
  const tokenIds = parseJsonishList(
    market.clobTokenIds || market.clob_token_ids || market.clobTokenIDs,
  );
  const upIndex = outcomes.indexOf("up");
  const downIndex = outcomes.indexOf("down");
  if (upIndex < 0 || downIndex < 0 || tokenIds.length <= Math.max(upIndex, downIndex)) {
    return null;
  }
  return {
    event_id: String(event.id || event.eventId || ""),
    market_id: String(market.id || ""),
    condition_id: String(market.conditionId || market.condition_id || ""),
    slug: String(event.slug || market.slug || ""),
    question: String(market.question || event.title || event.question || ""),
    start_epoch: startEpoch,
    end_epoch: startEpoch + intervalSeconds,
    tokens: {
      up: String(tokenIds[upIndex]),
      down: String(tokenIds[downIndex]),
    },
  };
}

async function currentMarket(config) {
  const poly = config.polymarket;
  const intervalSeconds = Number(config.market_interval_seconds || 300);
  const radius = Number(poly.slug_search_radius || 4);
  const nowSeconds = Date.now() / 1000;
  const baseStart = intervalStart(nowSeconds, intervalSeconds);
  const candidates = [];
  for (let offset = -radius; offset <= radius; offset += 1) {
    const start = baseStart + offset * intervalSeconds;
    const slug = String(poly.event_slug_template).replace("{start_epoch}", String(start));
    const url = `${poly.gamma_base_url}/events/slug/${slug}`;
    try {
      const response = await fetch(url, { headers: { "accept": "application/json" } });
      if (!response.ok) {
        continue;
      }
      const event = await response.json();
      const market = marketFromEvent(event, start, intervalSeconds);
      if (market && market.start_epoch <= nowSeconds && nowSeconds < market.end_epoch) {
        candidates.push(market);
      }
    } catch {
      continue;
    }
  }
  candidates.sort((left, right) => Math.abs(left.start_epoch - baseStart) - Math.abs(right.start_epoch - baseStart));
  return candidates[0] || null;
}

function eventTimestampMs(event) {
  const raw = event.timestamp || event.ts || event.time;
  if (raw === undefined || raw === null) {
    return null;
  }
  if (typeof raw === "number") {
    return raw > 10_000_000_000 ? raw : raw * 1000;
  }
  const parsedNumber = Number(raw);
  if (Number.isFinite(parsedNumber)) {
    return parsedNumber > 10_000_000_000 ? parsedNumber : parsedNumber * 1000;
  }
  const parsedDate = Date.parse(raw);
  return Number.isFinite(parsedDate) ? parsedDate : null;
}

function normalizeEvents(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }
  if (Array.isArray(payload.events)) {
    return payload.events;
  }
  return [payload];
}

function eventType(event) {
  return String(event.event_type || event.type || event.event || "unknown");
}

function assetId(event) {
  return String(event.asset_id || event.asset || event.token_id || event.token || "");
}

function bestBidAskSignature(event) {
  const asset = assetId(event);
  if (!asset) {
    return null;
  }
  const bestBid = event.best_bid || event.bestBid || event.bid || "";
  const bestAsk = event.best_ask || event.bestAsk || event.ask || "";
  const spread = event.spread || "";
  return `${asset}:${bestBid}:${bestAsk}:${spread}`;
}

function percentile(values, pct) {
  if (values.length === 0) {
    return 0;
  }
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil((pct / 100) * ordered.length) - 1),
  );
  return ordered[index];
}

function cadenceAdequacy({ config, durationSeconds, messages, events, interMessageMs, eventLagMs, bestBidAskChanges, streamError }) {
  const botSampleIntervalMs = Math.max(1, Number(config.sample_interval_seconds || 5) * 1000);
  const marketIntervalSeconds = Math.max(1, Number(config.market_interval_seconds || 300));
  const p95InterMessageMs = percentile(interMessageMs, 95);
  const maxInterMessageMs = interMessageMs.length ? Math.max(...interMessageMs) : 0;
  const p95EventLagMs = percentile(eventLagMs, 95);
  const maxEventLagMs = eventLagMs.length ? Math.max(...eventLagMs) : 0;
  const minEventLagMs = eventLagMs.length ? Math.min(...eventLagMs) : 0;
  const eventsPerSecond = durationSeconds > 0 ? events / durationSeconds : 0;
  const bestBidAskChangesPerMinute = durationSeconds > 0 ? bestBidAskChanges / (durationSeconds / 60) : 0;
  const interMessageWithinBotCadence = events > 0 && p95InterMessageMs <= botSampleIntervalMs;
  const maxGapWithinThreeBotSamples = events > 0 && maxInterMessageMs <= botSampleIntervalMs * 3;
  const timestampLagWithinBotCadence =
    eventLagMs.length > 0 && p95EventLagMs <= botSampleIntervalMs;
  let grade = "no_events";
  if (streamError) {
    grade = "stream_error";
  } else if (events > 0 && interMessageWithinBotCadence && timestampLagWithinBotCadence && maxGapWithinThreeBotSamples) {
    grade = "supports_bot_cadence";
  } else if (events > 0 && interMessageWithinBotCadence && timestampLagWithinBotCadence) {
    grade = "supports_bot_cadence_with_gaps";
  } else if (events > 0 && interMessageWithinBotCadence) {
    grade = "message_cadence_ok_timestamp_lag_unproven";
  } else if (events > 0) {
    grade = "below_bot_cadence";
  }
  const notes = [
    "This measures Polymarket CLOB stream freshness for subscribed token IDs; it does not prove fillability or queue position.",
    "The stream is event-driven, so a quiet interval can mean no visible market update rather than a polling failure.",
  ];
  if (eventLagMs.length === 0) {
    notes.push("No parseable event timestamps were observed, so timestamp-lag adequacy is unproven.");
  }
  if (minEventLagMs < 0) {
    notes.push("Some event timestamps were ahead of the local clock; treat timestamp lag as approximate because local clock skew or server timestamp semantics can create negative values.");
  }
  if (!interMessageWithinBotCadence && events > 0) {
    notes.push("P95 inter-message interval exceeded the configured bot polling interval.");
  }
  if (!maxGapWithinThreeBotSamples && events > 0) {
    notes.push("At least one observed stream gap exceeded three configured bot polling intervals.");
  }
  return {
    grade,
    bot_sample_interval_ms: botSampleIntervalMs,
    market_interval_seconds: marketIntervalSeconds,
    bot_samples_per_market: marketIntervalSeconds / (botSampleIntervalMs / 1000),
    events_per_second: eventsPerSecond,
    messages_per_second: durationSeconds > 0 ? messages / durationSeconds : 0,
    best_bid_ask_changes_per_minute: bestBidAskChangesPerMinute,
    p95_inter_message_ms: p95InterMessageMs,
    max_inter_message_ms: maxInterMessageMs,
    p95_event_lag_ms: p95EventLagMs,
    max_event_lag_ms: maxEventLagMs,
    min_event_lag_ms: minEventLagMs,
    timestamp_lag_samples: eventLagMs.length,
    inter_message_within_bot_cadence: interMessageWithinBotCadence,
    timestamp_lag_within_bot_cadence: timestampLagWithinBotCadence,
    max_gap_within_three_bot_samples: maxGapWithinThreeBotSamples,
    notes,
  };
}

function assertSelfTest(condition, message) {
  if (!condition) {
    throw new Error(`Self-test failed: ${message}`);
  }
}

function runSelfTest() {
  const config = { sample_interval_seconds: 5, market_interval_seconds: 300 };
  const fast = cadenceAdequacy({
    config,
    durationSeconds: 10,
    messages: 100,
    events: 100,
    interMessageMs: [5, 10, 20, 25, 30],
    eventLagMs: [40, 50, 60, 70],
    bestBidAskChanges: 12,
    streamError: null,
  });
  assertSelfTest(fast.grade === "supports_bot_cadence", "fast stream should support bot cadence");
  assertSelfTest(fast.bot_samples_per_market === 60, "bot samples per market should use config");
  const slow = cadenceAdequacy({
    config,
    durationSeconds: 10,
    messages: 5,
    events: 5,
    interMessageMs: [6000, 7000, 8000, 9000],
    eventLagMs: [40, 50, 60, 70],
    bestBidAskChanges: 1,
    streamError: null,
  });
  assertSelfTest(slow.grade === "below_bot_cadence", "slow stream should be below bot cadence");
  const noTimestamp = cadenceAdequacy({
    config,
    durationSeconds: 10,
    messages: 10,
    events: 10,
    interMessageMs: [5, 10, 20],
    eventLagMs: [],
    bestBidAskChanges: 1,
    streamError: null,
  });
  assertSelfTest(
    noTimestamp.grade === "message_cadence_ok_timestamp_lag_unproven",
    "missing timestamps should leave timestamp lag unproven",
  );
  console.log("websocket probe self-test passed");
}

function markdown(summary, rawPath) {
  const lines = [
    "# Polymarket WebSocket Data Probe",
    "",
    `Generated: \`${new Date().toISOString()}\``,
    `Raw events: \`${rawPath}\``,
    "",
    "## Summary",
    "",
    `- Market: \`${summary.market?.slug || "manual-assets"}\``,
    `- Assets subscribed: \`${summary.assets_subscribed.join(", ")}\``,
    `- Duration: \`${summary.duration_seconds.toFixed(1)}s\``,
    `- Messages: \`${summary.messages}\``,
    `- Stream events: \`${summary.events}\``,
    `- PONG heartbeats: \`${summary.pongs}\``,
    `- Event types: \`${JSON.stringify(summary.event_counts)}\``,
    `- Distinct assets seen: \`${summary.assets_seen.length}\``,
    `- Best bid/ask changes: \`${summary.best_bid_ask_changes}\``,
    `- Median inter-message interval: \`${summary.inter_message_ms.median.toFixed(1)}ms\``,
    `- P95 inter-message interval: \`${summary.inter_message_ms.p95.toFixed(1)}ms\``,
    `- Median event timestamp lag: \`${summary.event_lag_ms.median.toFixed(1)}ms\``,
    `- P95 event timestamp lag: \`${summary.event_lag_ms.p95.toFixed(1)}ms\``,
    "",
    "## Cadence Adequacy",
    "",
    `- Grade: \`${summary.cadence_adequacy.grade}\``,
    `- Bot sample interval: \`${summary.cadence_adequacy.bot_sample_interval_ms.toFixed(0)}ms\``,
    `- Bot samples per 5-minute market: \`${summary.cadence_adequacy.bot_samples_per_market.toFixed(1)}\``,
    `- Events per second: \`${summary.cadence_adequacy.events_per_second.toFixed(2)}\``,
    `- Best bid/ask changes per minute: \`${summary.cadence_adequacy.best_bid_ask_changes_per_minute.toFixed(2)}\``,
    `- P95 inter-message interval: \`${summary.cadence_adequacy.p95_inter_message_ms.toFixed(1)}ms\``,
    `- Max inter-message interval: \`${summary.cadence_adequacy.max_inter_message_ms.toFixed(1)}ms\``,
    `- P95 event timestamp lag: \`${summary.cadence_adequacy.p95_event_lag_ms.toFixed(1)}ms\``,
    `- Min event timestamp lag: \`${summary.cadence_adequacy.min_event_lag_ms.toFixed(1)}ms\``,
    `- Max event timestamp lag: \`${summary.cadence_adequacy.max_event_lag_ms.toFixed(1)}ms\``,
    "",
    "Cadence notes:",
    ...summary.cadence_adequacy.notes.map((note) => `- ${note}`),
    "",
    "## Notes",
    "",
    "- This probes Polymarket's market WebSocket stream, not the REST polling path.",
    "- `event timestamp lag` is only populated when the stream event includes a parseable timestamp.",
    "- The bot still uses REST snapshots for paper trading; this probe measures whether a lower-latency streaming path is usable.",
  ];
  if (summary.error) {
    lines.push("", "## Error", "", `\`${summary.error}\``);
  }
  return `${lines.join("\n")}\n`;
}

async function writeReports({ rows, summary, outputDir, stamp }) {
  await fsp.mkdir(outputDir, { recursive: true });
  const jsonPath = path.join(outputDir, `websocket_probe_${stamp}.json`);
  const mdPath = path.join(outputDir, `websocket_probe_${stamp}.md`);
  await fsp.writeFile(jsonPath, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  await fsp.writeFile(mdPath, markdown(summary, summary.raw_path), "utf8");
  console.log(JSON.stringify({ raw: summary.raw_path, json: jsonPath, markdown: mdPath, events: rows.length }, null, 2));
}

async function main() {
  const args = parseArgs(process.argv);
  if (args.selfTest) {
    runSelfTest();
    return;
  }
  if (typeof WebSocket !== "function") {
    throw new Error("This probe requires Node.js with a global WebSocket implementation.");
  }
  const config = await loadJson(args.config);
  const outputDir = args.outputDir;
  await fsp.mkdir(outputDir, { recursive: true });
  const stamp = utcStamp();
  const rawPath = path.join(outputDir, `websocket_probe_${stamp}.jsonl`);
  const raw = fs.createWriteStream(rawPath, { flags: "w" });
  const market = args.assets.length > 0 ? null : await currentMarket(config);
  const assets = args.assets.length > 0 ? args.assets : market ? [market.tokens.up, market.tokens.down] : [];
  if (assets.length === 0) {
    throw new Error("No active market assets found. Pass --asset-id TOKEN_ID to probe manually.");
  }

  const startedAt = Date.now();
  const rows = [];
  const eventCounts = {};
  const assetsSeen = new Set();
  const interMessageMs = [];
  const eventLagMs = [];
  let lastMessageAt = null;
  let lastBestBidAsk = new Map();
  let bestBidAskChanges = 0;
  let messages = 0;
  let pongs = 0;
  let streamError = null;

  const ws = new WebSocket(MARKET_WS_URL);
  const subscription = {
    type: "market",
    assets_ids: assets,
    custom_feature_enabled: true,
  };

  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send("PING");
    }
  }, 10_000);

  const done = new Promise((resolve) => {
    const stopTimer = setTimeout(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close(1000, "duration complete");
      }
    }, args.durationSeconds * 1000);

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify(subscription));
    });
    ws.addEventListener("message", (message) => {
      const receivedAt = Date.now();
      messages += 1;
      if (lastMessageAt !== null) {
        interMessageMs.push(receivedAt - lastMessageAt);
      }
      lastMessageAt = receivedAt;
      const text = String(message.data);
      if (text === "PONG") {
        pongs += 1;
        raw.write(`${JSON.stringify({ received_at: new Date(receivedAt).toISOString(), event_type: "PONG" })}\n`);
        return;
      }
      let payload;
      try {
        payload = JSON.parse(text);
      } catch {
        const row = {
          received_at: new Date(receivedAt).toISOString(),
          event_type: "raw_text",
          payload: text,
        };
        rows.push(row);
        raw.write(`${JSON.stringify(row)}\n`);
        return;
      }
      for (const event of normalizeEvents(payload)) {
        const type = eventType(event);
        const asset = assetId(event);
        if (asset) {
          assetsSeen.add(asset);
        }
        eventCounts[type] = (eventCounts[type] || 0) + 1;
        const eventMs = eventTimestampMs(event);
        if (eventMs !== null) {
          eventLagMs.push(receivedAt - eventMs);
        }
        const signature = bestBidAskSignature(event);
        if (signature) {
          const assetKey = signature.split(":")[0];
          if (lastBestBidAsk.get(assetKey) !== signature) {
            if (lastBestBidAsk.has(assetKey)) {
              bestBidAskChanges += 1;
            }
            lastBestBidAsk.set(assetKey, signature);
          }
        }
        const row = {
          received_at: new Date(receivedAt).toISOString(),
          received_epoch_ms: receivedAt,
          event_type: type,
          asset_id: asset || null,
          event_lag_ms: eventMs === null ? null : receivedAt - eventMs,
          payload: event,
        };
        rows.push(row);
        raw.write(`${JSON.stringify(row)}\n`);
      }
    });
    ws.addEventListener("error", (event) => {
      streamError = event.message || "WebSocket error";
    });
    ws.addEventListener("close", () => {
      clearTimeout(stopTimer);
      resolve();
    });
  });

  await done;
  clearInterval(heartbeat);
  raw.end();

  const elapsedSeconds = (Date.now() - startedAt) / 1000;
  const adequacy = cadenceAdequacy({
    config,
    durationSeconds: elapsedSeconds,
    messages,
    events: rows.length,
    interMessageMs,
    eventLagMs,
    bestBidAskChanges,
    streamError,
  });
  const summary = {
    generated_at: new Date().toISOString(),
    source: MARKET_WS_URL,
    raw_path: rawPath,
    market,
    subscription,
    assets_subscribed: assets,
    duration_seconds: elapsedSeconds,
    messages,
    events: rows.length,
    pongs,
    event_counts: eventCounts,
    assets_seen: [...assetsSeen].sort(),
    best_bid_ask_changes: bestBidAskChanges,
    inter_message_ms: {
      median: percentile(interMessageMs, 50),
      p95: percentile(interMessageMs, 95),
      max: interMessageMs.length ? Math.max(...interMessageMs) : 0,
    },
    event_lag_ms: {
      median: percentile(eventLagMs, 50),
      p95: percentile(eventLagMs, 95),
      max: eventLagMs.length ? Math.max(...eventLagMs) : 0,
      samples: eventLagMs.length,
    },
    cadence_adequacy: adequacy,
    error: streamError,
  };
  await writeReports({ rows, summary, outputDir, stamp });
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
