from __future__ import annotations

import html
import json
from typing import Any


def _json_script_payload(data: Any) -> str:
    """Embed JSON in a script tag without breaking JSON.parse."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def render_trace_dashboard(
    *,
    service_name: str,
    stats: dict[str, Any],
    traces: list[dict[str, Any]],
) -> str:
    stats_json = _json_script_payload(stats)
    traces_json = _json_script_payload(traces)
    title = html.escape(service_name)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} · Traces</title>
  <style>
    :root {{
      --bg: #faf9f5;
      --surface: #ffffff;
      --surface-muted: #f3f1eb;
      --border: #e8e6df;
      --text: #141413;
      --text-muted: #6b6962;
      --accent: #d97757;
      --accent-soft: #f4e8e1;
      --ok: #2f6f4e;
      --ok-soft: #e8f3ec;
      --error: #b54545;
      --error-soft: #fceeee;
      --shadow: 0 1px 2px rgba(20, 20, 19, 0.04), 0 8px 24px rgba(20, 20, 19, 0.06);
      --radius: 14px;
      --font: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono: ui-monospace, "SF Mono", "Cascadia Code", monospace;
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}

    .page {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 64px;
    }}

    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 28px;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}

    .mark {{
      width: 36px;
      height: 36px;
      border-radius: 10px;
      background: linear-gradient(145deg, #e8a48b, var(--accent));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.35);
    }}

    h1 {{
      margin: 0;
      font-size: 1.35rem;
      font-weight: 600;
      letter-spacing: -0.02em;
    }}

    .subtitle {{
      margin: 2px 0 0;
      color: var(--text-muted);
      font-size: 0.92rem;
    }}

    .actions {{
      display: flex;
      gap: 10px;
      align-items: center;
    }}

    button, .link-btn {{
      appearance: none;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 14px;
      font: inherit;
      font-size: 0.88rem;
      cursor: pointer;
      text-decoration: none;
      transition: background 0.15s ease, border-color 0.15s ease;
    }}

    button:hover, .link-btn:hover {{
      background: var(--surface-muted);
      border-color: #d8d5cc;
    }}

    button.primary {{
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }}

    button.primary:hover {{
      background: #c96849;
      border-color: #c96849;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 24px;
      overflow: visible;
    }}

    .stat {{
      position: relative;
      overflow: visible;
      border-radius: calc(var(--radius) + 2px);
      padding: 18px 18px 16px;
      cursor: help;
      border: 1px solid rgba(255, 255, 255, 0.35);
      box-shadow: var(--shadow);
      color: #fff;
      min-height: 118px;
      background: linear-gradient(145deg, #e8a87c 0%, #d97757 38%, #b8653a 72%, #9a4e28 100%);
      isolation: isolate;
    }}

    .stat:hover,
    .stat:focus-within {{
      z-index: 30;
    }}

    .stat::before {{
      content: "";
      position: absolute;
      inset: 0;
      opacity: 0.2;
      background: radial-gradient(circle at top right, rgba(255, 248, 240, 0.9), transparent 58%);
      pointer-events: none;
      border-radius: inherit;
    }}

    .stat-tip {{
      position: absolute;
      left: 50%;
      top: calc(100% + 10px);
      transform: translateX(-50%);
      width: max-content;
      max-width: 260px;
      padding: 8px 10px;
      border-radius: 8px;
      background: #141413;
      color: #fff;
      font-size: 0.78rem;
      line-height: 1.45;
      text-transform: none;
      letter-spacing: normal;
      font-weight: 500;
      white-space: normal;
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
      transition: opacity 0.15s ease, visibility 0.15s ease;
      z-index: 40;
      box-shadow: var(--shadow);
    }}

    .stat-tip::after {{
      content: "";
      position: absolute;
      bottom: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 6px solid transparent;
      border-bottom-color: #141413;
    }}

    .stat:hover .stat-tip,
    .stat:focus-within .stat-tip {{
      opacity: 1;
      visibility: visible;
    }}

    .stat-head {{
      position: relative;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }}

    .stat-label {{
      position: relative;
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      opacity: 0.88;
      margin-bottom: 0;
      font-weight: 700;
    }}

    .stat-percentile {{
      flex-shrink: 0;
      font-size: 0.72rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      padding: 3px 9px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.22);
      border: 1px solid rgba(255, 255, 255, 0.35);
    }}

    .stat-value {{
      position: relative;
      font-size: 1.85rem;
      font-weight: 800;
      letter-spacing: -0.04em;
      line-height: 1.05;
    }}

    .stat-sub {{
      position: relative;
      margin-top: 8px;
      font-size: 0.78rem;
      opacity: 0.86;
      font-weight: 600;
    }}

    .panel {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: calc(var(--radius) + 2px);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}

    .panel-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #fff, #fcfaf6);
    }}

    .panel-head h2 {{
      margin: 0;
      font-size: 1rem;
      font-weight: 600;
    }}

    .empty {{
      padding: 48px 24px;
      text-align: center;
      color: var(--text-muted);
    }}

    .trace-list {{
      display: flex;
      flex-direction: column;
    }}

    .trace {{
      border-bottom: 1px solid var(--border);
    }}

    .trace:last-child {{ border-bottom: none; }}

    .trace-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      padding: 16px 18px;
      cursor: pointer;
      transition: background 0.12s ease;
    }}

    .trace-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}

    .trace-row:hover {{ background: #fcfaf6; }}

    .trace-name {{
      font-weight: 600;
      letter-spacing: -0.01em;
    }}

    .trace-meta {{
      margin-top: 4px;
      font-size: 0.84rem;
      color: var(--text-muted);
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.78rem;
      font-weight: 600;
      white-space: nowrap;
    }}

    .pill.ok {{ background: var(--ok-soft); color: var(--ok); }}
    .pill.error {{ background: var(--error-soft); color: var(--error); }}
    .pill.blocked {{ background: #fce8dc; color: #9a4e28; }}
    .pill.neutral {{ background: var(--surface-muted); color: var(--text-muted); }}

    .trace-detail {{
      display: none;
      padding: 0 18px 18px;
      background: #fcfaf6;
    }}

    .trace.open .trace-detail {{ display: block; }}

    .span-tree {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}

    .span-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}

    .span-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px 14px;
      cursor: pointer;
      transition: background 0.12s ease;
    }}

    .span-row:hover {{
      background: #fcfaf6;
    }}

    .span-chevron {{
      color: var(--text-muted);
      font-size: 0.95rem;
      width: 12px;
      flex-shrink: 0;
      transition: transform 0.15s ease;
      user-select: none;
    }}

    .span-card.open > .span-row .span-chevron {{
      transform: rotate(90deg);
    }}

    .span-head {{
      flex: 1;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      min-width: 0;
    }}

    .span-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }}

    .span-title {{
      font-weight: 600;
      font-size: 0.95rem;
    }}

    .span-sub {{
      margin-top: 2px;
      color: var(--text-muted);
      font-size: 0.82rem;
      font-family: var(--mono);
    }}

    .span-body {{
      display: none;
      padding: 12px 14px 14px 36px;
      border-top: 1px solid var(--border);
      background: #fcfaf6;
    }}

    .span-card.open > .span-body {{
      display: block;
    }}

    .span-children {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin-top: 10px;
      padding-left: 10px;
      border-left: 2px solid var(--border);
    }}

    .attrs {{
      display: grid;
      gap: 6px;
    }}

    .attr {{
      display: grid;
      grid-template-columns: 180px 1fr;
      gap: 10px;
      font-size: 0.84rem;
      padding: 6px 0;
      border-top: 1px dashed var(--border);
    }}

    .attr:first-child {{ border-top: none; padding-top: 0; }}

    .attr-key {{
      color: var(--text-muted);
      font-family: var(--mono);
      word-break: break-word;
    }}

    .attr-val {{
      word-break: break-word;
    }}

    .footer-note {{
      margin-top: 18px;
      text-align: center;
      color: var(--text-muted);
      font-size: 0.82rem;
    }}

    .live-status {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.82rem;
      color: var(--text-muted);
    }}

    .live-dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--ok);
      box-shadow: 0 0 0 4px var(--ok-soft);
    }}

    .live-dot.reconnecting {{
      background: #c68a2e;
      box-shadow: 0 0 0 4px #f7ecd8;
    }}

    .live-dot.offline {{
      background: var(--error);
      box-shadow: 0 0 0 4px var(--error-soft);
    }}

    @media (max-width: 860px) {{
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .trace-row {{ grid-template-columns: 1fr; }}
      .attr {{ grid-template-columns: 1fr; }}
      .trace-pills {{ flex-wrap: wrap; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div class="brand">
        <div class="mark" aria-hidden="true"></div>
        <div>
          <h1>{title}</h1>
          <p class="subtitle">OpenTelemetry traces · in-memory store</p>
        </div>
      </div>
      <div class="actions">
        <span class="live-status" id="live-status">
          <span class="live-dot" id="live-dot" aria-hidden="true"></span>
          <span id="live-label">Connecting…</span>
        </span>
        <a class="link-btn" href="/v1/metrics/cost-latency">Cost API</a>
        <a class="link-btn" href="/v1/metrics/inference">Inference API</a>
        <button class="primary" id="refresh-btn" type="button">Refresh</button>
      </div>
    </header>

    <section class="stats" id="stats"></section>

    <section class="panel">
      <div class="panel-head">
        <h2>Recent traces</h2>
        <span class="pill neutral" id="trace-count">0 traces</span>
      </div>
      <div class="trace-list" id="trace-list"></div>
    </section>

    <p class="footer-note">Live updates via SSE. Traces reset when the Space restarts.</p>
  </div>

  <script id="bootstrap-stats" type="application/json">{stats_json}</script>
  <script id="bootstrap-traces" type="application/json">{traces_json}</script>
  <script>
    const statsEl = document.getElementById("stats");
    const listEl = document.getElementById("trace-list");
    const countEl = document.getElementById("trace-count");
    const refreshBtn = document.getElementById("refresh-btn");
    const liveDot = document.getElementById("live-dot");
    const liveLabel = document.getElementById("live-label");
    let stream = null;
    let reconnectTimer = null;

    function setLiveStatus(state, label) {{
      liveDot.className = "live-dot" + (state ? " " + state : "");
      liveLabel.textContent = label;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }}

    function shortId(id) {{
      if (!id) return "—";
      return id.length > 12 ? id.slice(0, 12) + "…" : id;
    }}

    function formatDurationSeconds(ms) {{
      const num = Number(ms);
      if (Number.isNaN(num)) return "—";
      return `${{(num / 1000).toFixed(2)}} s`;
    }}

    function formatMsAsSeconds(ms) {{
      return formatDurationSeconds(ms);
    }}

    function isIsoTimestamp(value) {{
      return typeof value === "string" && /^\\d{{4}}-\\d{{2}}-\\d{{2}}T/.test(value);
    }}

    function formatTimestampIST(iso) {{
      if (!iso) return "—";
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return String(iso);
      const datePart = new Intl.DateTimeFormat("en-GB", {{
        timeZone: "Asia/Kolkata",
        day: "2-digit",
        month: "short",
        year: "numeric",
      }}).format(date);
      const timePart = new Intl.DateTimeFormat("en-IN", {{
        timeZone: "Asia/Kolkata",
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit",
        hour12: true,
      }}).format(date);
      return `${{datePart}}, ${{timePart}} IST`;
    }}

    function formatDisplayValue(key, value) {{
      const keyLower = String(key).toLowerCase();
      if (
        keyLower.endsWith("_ms")
        || keyLower.includes("duration_ms")
        || keyLower === "latency_ms"
      ) {{
        const num = Number(value);
        if (!Number.isNaN(num)) return formatDurationSeconds(num);
      }}
      if (keyLower.includes("tokens_per_sec") || keyLower === "inference.tokens_per_sec") {{
        return formatTokensPerSec(value);
      }}
      if (isIsoTimestamp(value) || keyLower.endsWith("_time") || keyLower.endsWith(".time")) {{
        return formatTimestampIST(value);
      }}
      return value;
    }}

    function formatOptionalMs(ms) {{
      if (ms == null || ms === "") return "—";
      return formatMsAsSeconds(ms);
    }}

    function formatTokensPerSec(value) {{
      const num = Number(value);
      if (Number.isNaN(num) || num <= 0) return "—";
      return `${{num.toFixed(1)}} tok/s`;
    }}

    function renderStatCard(metric, percentile, value, tip, sub) {{
      const percentileHtml = percentile
        ? `<span class="stat-percentile">${{escapeHtml(percentile)}}</span>`
        : "";
      const tipHtml = tip
        ? `<div class="stat-tip" role="tooltip">${{escapeHtml(tip)}}</div>`
        : "";
      return `
        <div class="stat" tabindex="0">
          ${{tipHtml}}
          <div class="stat-head">
            <div class="stat-label">${{escapeHtml(metric)}}</div>
            ${{percentileHtml}}
          </div>
          <div class="stat-value">${{escapeHtml(value)}}</div>
          ${{sub ? `<div class="stat-sub">${{escapeHtml(sub)}}</div>` : ""}}
        </div>
      `;
    }}

    function formatUsd(value) {{
      const num = Number(value);
      if (Number.isNaN(num)) return "—";
      if (num >= 0.01) return `$${{num.toFixed(4)}}`;
      return `$${{num.toFixed(6)}}`;
    }}

    function renderStats(stats) {{
      const inference = stats.inference || {{}};
      const apiCost = stats.cost_latency || {{}};
      const consumed = Number(apiCost.api_cost_consumed_usd ?? 0);
      const requestCount = apiCost.total_requests ?? 0;
      const cards = [
        [
          "API $ Consumed",
          null,
          formatUsd(consumed),
          apiCost.estimate_tooltip || "Sum of estimated runtime + token proxy cost for all recorded requests.",
          requestCount ? `${{requestCount}} request${{requestCount === 1 ? "" : "s"}}` : "No requests yet",
        ],
        [
          "Total traces",
          null,
          String(stats.total_traces ?? 0),
          "Number of grouped request flows stored in memory.",
          "",
        ],
        [
          "TTFT",
          null,
          formatOptionalMs(inference.ttft_ms),
          "Time to first streamed token or chunk on the most recent request.",
          "Last request",
        ],
        [
          "TBT",
          null,
          formatOptionalMs(inference.tbt_ms),
          "Average gap between consecutive streamed chunks on the most recent request.",
          "Last request",
        ],
        [
          "Tokens/s",
          null,
          formatTokensPerSec(inference.tokens_per_sec),
          "Output tokens generated per second during decode on the most recent request.",
          "Inference speed",
        ],
        [
          "Latency",
          "p50",
          formatMsAsSeconds(inference.latency_p50_ms ?? 0),
          "Median end-to-end model latency across stored requests.",
          inference.samples ? `${{inference.samples}} samples` : "No samples yet",
        ],
        [
          "Latency",
          "p95",
          formatMsAsSeconds(inference.latency_p95_ms ?? 0),
          "95th percentile end-to-end model latency across stored requests.",
          "",
        ],
        [
          "Tokens",
          null,
          `${{Math.round(inference.avg_input_tokens ?? 0)}} in / ${{Math.round(inference.avg_output_tokens ?? 0)}} out`,
          "Average prompt and completion token counts using the model tokenizer.",
          "Per request",
        ],
      ];
      statsEl.innerHTML = cards.map(([metric, percentile, value, tip, sub]) =>
        renderStatCard(metric, percentile, value, tip, sub)
      ).join("");
    }}

    function renderAttributes(attrs) {{
      const entries = Object.entries(attrs || {{}});
      if (!entries.length) {{
        return '<div class="attr"><span class="attr-key">note</span><span class="attr-val">No attributes</span></div>';
      }}
      return entries.map(([key, value]) => `
        <div class="attr">
          <span class="attr-key">${{escapeHtml(key)}}</span>
          <span class="attr-val">${{escapeHtml(formatDisplayValue(key, value))}}</span>
        </div>
      `).join("");
    }}

    function buildSpanTree(spans) {{
      const nodes = (spans || []).map((span) => ({{
        ...span,
        children: [],
      }}));
      const byId = new Map(nodes.map((node) => [node.span_id, node]));
      const roots = [];

      for (const node of nodes) {{
        const parent = node.parent_span_id ? byId.get(node.parent_span_id) : null;
        if (parent) {{
          parent.children.push(node);
        }} else {{
          roots.push(node);
        }}
      }}

      const sortNodes = (items) => {{
        items.sort((a, b) => String(a.start_time || "").localeCompare(String(b.start_time || "")));
        items.forEach((item) => sortNodes(item.children));
      }};
      sortNodes(roots);
      return roots;
    }}

    function statusPillClass(status) {{
      if (status === "error") return "error";
      if (status === "blocked") return "blocked";
      return "ok";
    }}

    function formatTraceStatus(status) {{
      if (status === "blocked") return "blocked";
      return status || "ok";
    }}

    function renderSpanNode(span, openSpanIds) {{
      const isOpen = openSpanIds.has(span.span_id);
      const hasChildren = (span.children || []).length > 0;
      const hasAttrs = Object.keys(span.attributes || {{}}).length > 0;
      const childHtml = hasChildren
        ? `<div class="span-children">${{span.children.map((child) => renderSpanNode(child, openSpanIds)).join("")}}</div>`
        : "";

      return `
        <article class="span-card${{isOpen ? " open" : ""}}" data-span-id="${{escapeHtml(span.span_id)}}">
          <div class="span-row" role="button" tabindex="0" aria-expanded="${{isOpen ? "true" : "false"}}">
            <span class="span-chevron" aria-hidden="true">›</span>
            <div class="span-head">
              <div>
                <div class="span-title">${{escapeHtml(span.name)}}</div>
                <div class="span-sub">${{escapeHtml(span.kind)}} · ${{shortId(span.span_id)}}</div>
              </div>
              <div class="span-pills">
                <span class="pill neutral">${{escapeHtml(formatDurationSeconds(span.duration_ms))}}</span>
                ${{(span.status && span.status !== "ok") ? `<span class="pill ${{statusPillClass(span.status)}}">${{escapeHtml(formatTraceStatus(span.status))}}</span>` : ""}}
              </div>
            </div>
          </div>
          <div class="span-body">
            ${{hasAttrs ? `<div class="attrs">${{renderAttributes(span.attributes)}}</div>` : ""}}
            ${{!hasAttrs && !hasChildren ? '<div class="attrs"><div class="attr"><span class="attr-key">note</span><span class="attr-val">No details</span></div></div>' : ""}}
            ${{childHtml}}
          </div>
        </article>
      `;
    }}

    function renderSpanTree(spans, openSpanIds) {{
      return buildSpanTree(spans).map((span) => renderSpanNode(span, openSpanIds)).join("");
    }}

    function renderTraceInferencePills(trace) {{
      const pills = [
        `<span class="pill neutral">${{trace.span_count}} spans</span>`,
        `<span class="pill neutral">${{escapeHtml(formatDurationSeconds(trace.total_duration_ms))}}</span>`,
        `<span class="pill neutral">${{trace.input_tokens || 0}} in / ${{trace.output_tokens || 0}} out tok</span>`,
      ];
      if (trace.ttft_ms != null && trace.ttft_ms !== "") {{
        pills.push(`<span class="pill neutral">TTFT ${{escapeHtml(formatDurationSeconds(trace.ttft_ms))}}</span>`);
      }}
      if (trace.tbt_ms != null && trace.tbt_ms !== "") {{
        pills.push(`<span class="pill neutral">TBT ${{escapeHtml(formatDurationSeconds(trace.tbt_ms))}}</span>`);
      }}
      if (trace.tokens_per_sec != null && trace.tokens_per_sec !== "") {{
        pills.push(`<span class="pill neutral">${{escapeHtml(formatTokensPerSec(trace.tokens_per_sec))}}</span>`);
      }}
      for (const layer of trace.guardrail_blocks || []) {{
        pills.push(`<span class="pill blocked">${{escapeHtml(layer.replaceAll("_", " "))}}</span>`);
      }}
      pills.push(`<span class="pill ${{statusPillClass(trace.status)}}">${{escapeHtml(formatTraceStatus(trace.status))}}</span>`);
      return `<div class="trace-pills">${{pills.join("")}}</div>`;
    }}

    function renderTraces(traces) {{
      const openTraceIds = new Set(
        [...listEl.querySelectorAll(".trace.open")].map((node) => node.dataset.traceId)
      );
      const openSpanIds = new Set(
        [...listEl.querySelectorAll(".span-card.open")].map((node) => node.dataset.spanId)
      );

      countEl.textContent = `${{traces.length}} trace${{traces.length === 1 ? "" : "s"}}`;
      if (!traces.length) {{
        listEl.innerHTML = '<div class="empty">No traces yet. Send a chat request to populate this view.</div>';
        return;
      }}

      listEl.innerHTML = traces.map((trace, index) => `
        <section class="trace${{openTraceIds.has(trace.trace_id) ? " open" : ""}}" data-trace-id="${{escapeHtml(trace.trace_id)}}" data-trace-index="${{index}}">
          <div class="trace-row" role="button" tabindex="0" aria-expanded="${{openTraceIds.has(trace.trace_id) ? "true" : "false"}}">
            <div>
              <div class="trace-name">${{escapeHtml(trace.root_span)}}</div>
              <div class="trace-meta">${{shortId(trace.trace_id)}} · ${{escapeHtml(trace.service_name)}} · ${{escapeHtml(formatTimestampIST(trace.start_time))}}</div>
            </div>
            ${{renderTraceInferencePills(trace)}}
          </div>
          <div class="trace-detail">
            <div class="span-tree">
              ${{renderSpanTree(trace.spans || [], openSpanIds)}}
            </div>
          </div>
        </section>
      `).join("");
    }}

    listEl.addEventListener("click", (event) => {{
      const traceRow = event.target.closest(".trace-row");
      if (traceRow && listEl.contains(traceRow)) {{
        const trace = traceRow.closest(".trace");
        const open = trace.classList.toggle("open");
        traceRow.setAttribute("aria-expanded", open ? "true" : "false");
        return;
      }}

      const spanRow = event.target.closest(".span-row");
      if (spanRow && listEl.contains(spanRow)) {{
        event.stopPropagation();
        const card = spanRow.closest(".span-card");
        const open = card.classList.toggle("open");
        spanRow.setAttribute("aria-expanded", open ? "true" : "false");
      }}
    }});

    function applyPayload(payload) {{
      renderStats(payload.stats || {{}});
      renderTraces(payload.traces || []);
    }}

    async function loadData() {{
      refreshBtn.disabled = true;
      refreshBtn.textContent = "Refreshing…";
      try {{
        const response = await fetch("/v1/traces?limit=40");
        const payload = await response.json();
        applyPayload(payload);
      }} catch (error) {{
        listEl.innerHTML = '<div class="empty">Could not load traces.</div>';
        setLiveStatus("offline", "Offline");
      }} finally {{
        refreshBtn.disabled = false;
        refreshBtn.textContent = "Refresh";
      }}
    }}

    function connectStream() {{
      if (stream) {{
        stream.close();
        stream = null;
      }}
      if (reconnectTimer) {{
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }}

      setLiveStatus("reconnecting", "Connecting…");
      stream = new EventSource("/v1/traces/stream?limit=40");

      stream.addEventListener("open", () => {{
        setLiveStatus("", "Live");
      }});

      stream.addEventListener("snapshot", (event) => {{
        applyPayload(JSON.parse(event.data));
        setLiveStatus("", "Live");
      }});

      stream.addEventListener("update", (event) => {{
        applyPayload(JSON.parse(event.data));
        setLiveStatus("", "Live");
      }});

      stream.onerror = () => {{
        setLiveStatus("offline", "Reconnecting…");
        stream.close();
        stream = null;
        reconnectTimer = setTimeout(connectStream, 3000);
      }};
    }}

    try {{
      const initialStats = JSON.parse(document.getElementById("bootstrap-stats").textContent);
      const initialTraces = JSON.parse(document.getElementById("bootstrap-traces").textContent);
      applyPayload({{ stats: initialStats, traces: initialTraces }});
    }} catch (error) {{
      console.error("Failed to bootstrap trace UI:", error);
      loadData();
    }}

    refreshBtn.addEventListener("click", loadData);
    connectStream();
    window.addEventListener("beforeunload", () => {{
      if (stream) stream.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    }});
  </script>
</body>
</html>"""
