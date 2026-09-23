/**
 * Ollama control board — inline in Settings → LLMs → Ollama (live LEFT).
 * Host hook ducky:llm-slot. No DOM MutationObserver. Not a modal.
 */
(function () {
  "use strict";

  var PLUGIN_ID = "ollama";
  var MOUNT_ID = "ollama-library-mount";
  var STYLE_ID = "ollama-library-style";
  var CAPS = [
    { id: "vision", label: "Vision", icon: "eye" },
    { id: "tools", label: "Tools", icon: "wrench" },
    { id: "thinking", label: "Think", icon: "brain" },
    { id: "embedding", label: "Embed", icon: "layers" },
    { id: "cloud", label: "Cloud", icon: "cloud" },
  ];
  var state = {
    tab: "yours",
    local: null,
    catalog: [],
    query: "",
    caps: ["vision", "tools", "thinking"],
    order: "popular",
    selected: "",
    settings: null,
    pull: null,
    deleteName: "",
    busy: false,
    error: "",
    live: null,
    samples: [],
    history: [],
    histIndex: 4,
    boardOpen: false,
  };
  var pullTimer = 0;
  var liveTimer = 0;
  var localRetry = 0;
  var pickerMount = null;

  function panelApi() {
    try {
      return (window.pywebview && window.pywebview.api) || null;
    } catch (e) {
      return null;
    }
  }

  function call(method, params) {
    var api = panelApi();
    if (!api || typeof api.plugin_call !== "function") {
      return Promise.resolve({ ok: false, error: "Panel API not ready" });
    }
    return Promise.resolve(api.plugin_call(PLUGIN_ID, method, params || {}));
  }

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function icon(name) {
    var p =
      name === "eye"
        ? '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8S1 12 1 12z"/><circle cx="12" cy="12" r="3"/>'
        : name === "wrench"
          ? '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4L15 7z"/>'
          : name === "brain"
            ? '<path d="M9.5 2a3.5 3.5 0 0 0-3.3 4.6A3.5 3.5 0 0 0 4 9.5V14a3 3 0 0 0 3 3h1v3h4v-4h1a3 3 0 0 0 3-3V9.5a3.5 3.5 0 0 0-2.2-3A3.5 3.5 0 0 0 14.5 2 3.5 3.5 0 0 0 12 3.4 3.5 3.5 0 0 0 9.5 2z"/>'
            : name === "layers"
              ? '<path d="M12 2 2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>'
              : name === "cloud"
                ? '<path d="M18 10h-1.3A5 5 0 0 0 7 9a4 4 0 0 0 0 8h11a3 3 0 0 0 0-6z"/>'
                : name === "search"
                  ? '<circle cx="11" cy="11" r="7"/><path d="m20 20-3-3"/>'
                  : name === "board"
                    ? '<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>'
                    : '<circle cx="12" cy="12" r="4"/>';
    return (
      '<svg class="ollama-ico" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
      p +
      "</svg>"
    );
  }

  function heatColor(t) {
    var n = Math.max(0, Math.min(1, t));
    if (n <= 0.5) return "color-mix(in srgb, var(--amber) " + Math.round(n * 200) + "%, var(--green))";
    return "color-mix(in srgb, var(--red) " + Math.round((n - 0.5) * 200) + "%, var(--amber))";
  }

  function heatOf(val, lo, hi) {
    var a = Number(lo);
    var b = Number(hi);
    if (b === a) return 0;
    return Math.max(0, Math.min(1, (Number(val) - a) / (b - a)));
  }

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = [
      "#ollama-library-mount{margin-top:12px;min-height:520px}",
      ".ollama-ico{flex-shrink:0}",
      ".ollama-chip{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--border);background:var(--hover);color:var(--muted);border-radius:999px;padding:3px 9px;font-size:11px;line-height:1.3;cursor:pointer;font:inherit}",
      ".ollama-chip.is-on{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--text)}",
      ".ollama-filters{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}",
      ".ollama-search{display:flex;align-items:center;gap:8px}",
      ".ollama-search .settings-input{flex:1}",
      ".ollama-table{display:flex;flex-direction:column}",
      ".ollama-tr{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:12px;align-items:center;padding:10px 2px;border-top:1px solid var(--border)}",
      ".ollama-tr.is-selected{background:color-mix(in srgb,var(--accent) 10%,transparent);margin:0 -8px;padding:10px 8px;border-radius:8px}",
      ".ollama-tr-main{min-width:0;text-align:left;background:none;border:0;padding:0;color:inherit;cursor:pointer;font:inherit}",
      ".ollama-tr-title{display:flex;align-items:center;gap:8px;font-weight:600;color:var(--text)}",
      ".ollama-tr-meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:4px;color:var(--muted);font-size:12px}",
      ".ollama-tr-caps{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}",
      ".ollama-board{width:100%;height:min(680px,calc(100vh - 220px));display:flex;flex-direction:column;background:color-mix(in srgb,var(--card) 70%,var(--bg));border:1px solid var(--border);border-radius:12px;overflow:hidden;color:var(--text)}",
      ".ollama-board-head{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--border)}",
      ".ollama-board-head h3{margin:0;flex:1;font-size:14px}",
      ".ollama-board-main{min-height:0;flex:1;overflow:auto;padding:12px}",
      ".ollama-row-actions{display:flex;gap:6px;flex-shrink:0}",
      ".ollama-faders{display:grid;grid-template-columns:repeat(auto-fill,minmax(76px,1fr));gap:8px;margin:10px 0}",
      ".ollama-fader{display:flex;flex-direction:column;align-items:center;gap:6px;padding:10px 6px 8px;border:1px solid var(--border);border-radius:10px;background:color-mix(in srgb,var(--card) 80%,var(--bg))}",
      ".ollama-fader-label{font-size:10px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);text-align:center}",
      ".ollama-fader-val{font-size:12px;font-variant-numeric:tabular-nums;color:var(--text)}",
      ".ollama-fader-track{position:relative;width:28px;height:96px;border-radius:8px;background:var(--bg-elevated,var(--card));box-shadow:inset 0 2px 6px color-mix(in srgb,var(--bg) 80%,transparent);overflow:hidden}",
      ".ollama-fader-dots{position:absolute;inset:0;background-image:radial-gradient(circle,var(--muted) 1.2px,transparent 1.2px);background-size:8px 8px}",
      ".ollama-fader-fill{position:absolute;left:0;right:0;bottom:0;height:calc(100% * var(--effort-heat,0));background-image:radial-gradient(circle,var(--slider-color,var(--green)) 2px,transparent 2px);background-size:8px 8px}",
      ".ollama-fader-track input{position:absolute;inset:0;opacity:0.02;width:100%;height:100%;margin:0;cursor:pointer;writing-mode:vertical-lr;direction:rtl}",
      ".ollama-keep{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px}",
      ".ollama-live-status{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-bottom:10px}",
      ".ollama-graph{margin:0 0 12px}",
      ".ollama-graph-head{display:flex;justify-content:space-between;gap:8px;font-size:11px;margin-bottom:4px}",
      ".ollama-graph-head span:last-child{color:var(--muted);font-variant-numeric:tabular-nums}",
      ".ollama-svg{width:100%;height:48px;display:block}",
      ".ollama-split{display:grid;grid-template-columns:minmax(0,1fr) minmax(240px,.9fr);gap:16px;align-items:start}",
      ".ollama-board p,.ollama-board a{color:inherit;text-decoration:none}",
      ".ollama-error{margin:0 0 8px;color:var(--red,#f87171);font-size:13px;line-height:1.4}",
      ".ollama-empty{margin:0;color:var(--muted);font-size:13px;line-height:1.45}",
      ".ollama-picker{padding:8px 10px 4px}",
      ".ollama-picker .ollama-faders{grid-template-columns:repeat(4,minmax(0,1fr))}",
    ].join("");
    document.head.appendChild(el);
  }

  function ollamaSlide() {
    return document.querySelector('[data-ducky-llm-slot="settings"][data-provider="ollama"]');
  }

  function bareModel(id) {
    var s = String(id || "").trim();
    if (s.toLowerCase().indexOf("ollama:") === 0) s = s.slice(7);
    return s;
  }

  function isOllamaDetail(d) {
    var a = String((d && d.providerId) || "")
      .toLowerCase()
      .replace(/-/g, "_");
    var b = String((d && d.pluginId) || "")
      .toLowerCase()
      .replace(/-/g, "_");
    return a === "ollama" || b === "ollama";
  }

  function inOllamaUi(el) {
    var settings = document.getElementById(MOUNT_ID);
    if (settings && settings.contains(el)) return true;
    return !!(pickerMount && pickerMount.contains(el));
  }

  function faderHtml(opts) {
    var heat = heatOf(opts.value, opts.lo, opts.hi);
    return (
      '<div class="ollama-fader" style="--effort-heat:' +
      heat +
      ";--slider-color:" +
      heatColor(heat) +
      '"><span class="ollama-fader-label" data-no-translate>' +
      esc(opts.label) +
      '</span><div class="ollama-fader-track"><span class="ollama-fader-dots"></span><span class="ollama-fader-fill"></span>' +
      '<input type="range" data-act="' +
      esc(opts.act) +
      '"' +
      (opts.opt ? ' data-opt="' + esc(opts.opt) + '"' : "") +
      ' min="' +
      opts.lo +
      '" max="' +
      opts.hi +
      '" step="' +
      (opts.step == null ? 1 : opts.step) +
      '" value="' +
      opts.value +
      '" /></div><span class="ollama-fader-val">' +
      esc(opts.readout) +
      "</span></div>"
    );
  }

  function areaChart(values, colorVar, ceiling) {
    var pts = values || [];
    if (!pts.length) return '<svg class="ollama-svg" viewBox="0 0 100 40"></svg>';
    var max = ceiling || Math.max.apply(null, pts.concat([1]));
    var w = 100;
    var step = pts.length > 1 ? w / (pts.length - 1) : w;
    var line = pts
      .map(function (v, i) {
        return (i * step).toFixed(1) + "," + (38 - (Number(v) / max) * 34).toFixed(1);
      })
      .join(" ");
    var fill =
      "0,40 " +
      pts
        .map(function (v, i) {
          return (i * step).toFixed(1) + "," + (38 - (Number(v) / max) * 34).toFixed(1);
        })
        .join(" ") +
      " 100,40";
    var c = colorVar || "--accent";
    return (
      '<svg class="ollama-svg" viewBox="0 0 100 40" preserveAspectRatio="none">' +
      '<polygon fill="color-mix(in srgb, var(' +
      c +
      ') 28%, transparent)" points="' +
      fill +
      '" />' +
      '<polyline fill="none" stroke="var(' +
      c +
      ')" stroke-width="1.6" stroke-linejoin="round" points="' +
      line +
      '" /></svg>'
    );
  }

  function graphCard(label, readout, pts, colorVar, ceiling) {
    return (
      '<div class="ollama-graph"><div class="ollama-graph-head"><span data-no-translate>' +
      esc(label) +
      "</span><span>" +
      esc(readout) +
      "</span></div>" +
      areaChart(pts, colorVar, ceiling) +
      "</div>"
    );
  }

  function ctxFaders(st) {
    var ticks = (st && st.ticks) || [];
    var labels = (st && st.tick_labels) || [];
    var idx = Math.max(0, ticks.indexOf(st.num_ctx));
    if (ticks.indexOf(st.num_ctx) < 0) idx = Math.max(0, ticks.length - 1);
    return faderHtml({
      label: "Context",
      act: "ctx",
      lo: 0,
      hi: Math.max(0, ticks.length - 1),
      step: 1,
      value: idx,
      readout: labels[idx] || st.num_ctx || "",
    });
  }

  function optFaders(st, limit) {
    var opts = (st && st.options) || {};
    var sliders = (st && st.sliders) || [];
    if (limit) sliders = sliders.slice(0, limit);
    return sliders
      .map(function (s) {
        var val = opts[s.id] != null ? opts[s.id] : s.default;
        return faderHtml({
          label: s.label,
          act: "opt",
          opt: s.id,
          lo: s.lo,
          hi: s.hi,
          step: s.step,
          value: val,
          readout: String(val),
        });
      })
      .join("");
  }

  function switchRow(st) {
    var opts = (st && st.options) || {};
    var switches = (st && st.switches) || [];
    return (
      '<div class="ollama-keep">' +
      '<button type="button" class="settings-btn' +
      (st.keep_alive === -1 ? " is-on" : "") +
      '" data-act="alive" data-value="-1">Stay loaded</button>' +
      '<button type="button" class="settings-btn' +
      (st.keep_alive === 900 ? " is-on" : "") +
      '" data-act="alive" data-value="900">15m</button>' +
      '<button type="button" class="settings-btn' +
      (st.keep_alive === 0 ? " is-on" : "") +
      '" data-act="alive" data-value="0">Unload idle</button>' +
      '<button type="button" class="settings-btn" data-act="reset">Reset</button>' +
      switches
        .map(function (s) {
          var on = !!(opts[s.id] != null ? opts[s.id] : s.default);
          return (
            '<button type="button" class="settings-btn' +
            (on ? " is-on" : "") +
            '" data-act="switch" data-opt="' +
            esc(s.id) +
            '">' +
            esc(s.label) +
            "</button>"
          );
        })
        .join("") +
      "</div>"
    );
  }

  function tuneBoard(st) {
    if (!state.selected || !st || !st.ok) {
      return '<p class="ollama-empty" data-no-translate>Select a model on the right to open the mixer.</p>';
    }
    return (
      '<div class="ollama-faders">' +
      ctxFaders(st) +
      optFaders(st) +
      "</div>" +
      switchRow(st)
    );
  }

  function livePane() {
    var live = state.live || {};
    var samples = state.samples || [];
    var series = function (key) {
      return samples.map(function (s) {
        return s[key] == null ? 0 : s[key];
      });
    };
    var runNote = live.thinking
      ? "Thinking · " + (live.generating_model || "")
      : live.running
        ? "Warm"
        : "Idle";
    var vramPct =
      live.vram_total ? (100 * (live.vram_used || 0)) / live.vram_total : live.gpu_pct || 0;
    var calls = state.history || [];
    var idx = Math.max(0, Math.min(calls.length - 1, state.histIndex));
    var call = calls[idx];
    var loaded = (live.models || [])
      .map(function (m) {
        return (
          '<div class="ollama-tr-meta"><span data-no-translate>' +
          esc(m.name) +
          "</span><span>" +
          esc(m.size_vram_label || "") +
          " VRAM</span></div>"
        );
      })
      .join("");
    var hist = "";
    if (calls.length) {
      hist =
        '<div class="ollama-graph-head"><span>History</span><span>' +
        (idx + 1) +
        " / " +
        calls.length +
        "</span></div>" +
        faderHtml({
          label: "Call",
          act: "hist",
          lo: 0,
          hi: calls.length - 1,
          step: 1,
          value: idx,
          readout: String((call && call.tokens_per_s) || 0) + " tok/s",
        }) +
        '<p class="general-tab-section-desc" data-no-translate>' +
        esc((call && call.model) || "") +
        " · " +
        ((call && call.tokens_per_s) || 0) +
        " tok/s · " +
        ((call && call.elapsed_s) || 0) +
        "s · ctx " +
        ((call && call.num_ctx) || "?") +
        "</p>" +
        areaChart(
          calls.map(function (c) {
            return c.tokens_per_s || 0;
          }),
          "--green"
        ) +
        '<button type="button" class="settings-btn" data-act="apply-hist" data-index="' +
        idx +
        '">Apply sliders</button>';
    }
    return (
      '<div class="ollama-live-status"><span>' +
      esc(runNote) +
      "</span><strong>" +
      (live.gpu_name ? esc(live.gpu_name) : "PC") +
      "</strong></div>" +
      graphCard("CPU" + (live.cpu_count ? " · " + live.cpu_count + "c" : ""), (live.cpu_pct != null ? live.cpu_pct + "%" : "n/a"), series("cpu_pct"), "--accent", 100) +
      graphCard("RAM", live.ram_label || "n/a", series("ram_pct"), "--amber", 100) +
      graphCard("GPU", live.gpu_pct != null ? live.gpu_pct + "%" : "n/a", series("gpu_pct"), "--green", 100) +
      graphCard("VRAM", live.vram_label || "n/a", series("vram_pct").length ? series("vram_pct") : samples.map(function () { return vramPct; }), "--red", 100) +
      graphCard("Disk", live.disk_label || "n/a", series("disk_pct"), "--blue", 100) +
      (live.gpu_temp != null ? graphCard("GPU temp", live.gpu_temp + "°C", series("gpu_temp"), "--amber", 100) : "") +
      (live.gpu_power != null
        ? graphCard(
            "Power",
            live.gpu_power + " W" + (live.gpu_power_limit ? " / " + live.gpu_power_limit : ""),
            series("gpu_power"),
            "--accent"
          )
        : "") +
      (live.gpu_fan != null ? graphCard("Fan", live.gpu_fan + "%", series("gpu_fan"), "--muted", 100) : "") +
      (live.gpu_clock != null ? '<div class="ollama-tr-meta"><span>Clock</span><span>' + live.gpu_clock + " MHz</span></div>" : "") +
      (loaded || '<p class="general-tab-section-desc">No model in VRAM.</p>') +
      hist
    );
  }

  function yoursPane() {
    var local = state.local || {};
    var models = local.models || [];
    var rows = models
      .map(function (m) {
        var caps = (m.capabilities || []).map(function (c) {
          return String(c).toLowerCase();
        });
        var sel = m.name === state.selected ? " is-selected" : "";
        return (
          '<div class="ollama-tr' +
          sel +
          '"><button type="button" class="ollama-tr-main" data-act="select" data-name="' +
          esc(m.name) +
          '"><div class="ollama-tr-title" data-no-translate><span>' +
          esc(m.name) +
          "</span>" +
          (m.loaded ? '<span class="ollama-chip is-on">Loaded</span>' : "") +
          '</div><div class="ollama-tr-meta"><span>' +
          esc(m.size_label || "") +
          "</span>" +
          (m.parameter_size ? "<span>" + esc(m.parameter_size) + "</span>" : "") +
          (m.quantization ? "<span>" + esc(m.quantization) + "</span>" : "") +
          (m.context_length ? "<span>" + Math.round(m.context_length / 1024) + "k ctx</span>" : "") +
          '</div><div class="ollama-tr-caps">' +
          ["vision", "tools", "thinking"]
            .filter(function (c) {
              return caps.indexOf(c) >= 0;
            })
            .map(function (c) {
              return '<span class="ollama-chip is-on" data-no-translate>' + icon(c === "vision" ? "eye" : c === "tools" ? "wrench" : "brain") + " " + c + "</span>";
            })
            .join("") +
          '</div></button><div class="ollama-row-actions">' +
          (state.deleteName === m.name
            ? '<button type="button" class="settings-btn" data-act="delete-no">Cancel</button><button type="button" class="settings-btn" data-act="delete-yes" data-name="' +
              esc(m.name) +
              '">Delete</button>'
            : '<button type="button" class="settings-btn" data-act="delete" data-name="' +
              esc(m.name) +
              '">Delete</button>') +
          "</div></div>"
        );
      })
      .join("");
    return (
      '<div class="ollama-split"><div class="ollama-split-mix">' +
      tuneBoard(state.settings) +
      '</div><div class="ollama-split-list">' +
      (local.error ? '<p class="ollama-error" data-no-translate>' + esc(local.error) + "</p>" : "") +
      '<div class="ollama-table">' +
      (rows || '<p class="ollama-empty" data-no-translate>Nothing pulled yet. Open Library to download.</p>') +
      "</div></div></div>"
    );
  }

  function libraryPane() {
    var installed = {};
    ((state.local && state.local.models) || []).forEach(function (m) {
      installed[String(m.name || "").split(":")[0]] = true;
    });
    var chips = CAPS.map(function (c) {
      return (
        '<button type="button" class="ollama-chip' +
        (state.caps.indexOf(c.id) >= 0 ? " is-on" : "") +
        '" data-act="cap" data-name="' +
        c.id +
        '" data-no-translate title="' +
        c.label +
        '">' +
        icon(c.icon) +
        " " +
        c.label +
        "</button>"
      );
    }).join("");
    var cards = (state.catalog || [])
      .map(function (row) {
        var slug = row.slug || "";
        var have = installed[slug.split("/").pop()];
        return (
          '<div class="ollama-tr"><div class="ollama-tr-main"><div class="ollama-tr-title" data-no-translate><span>' +
          esc(row.name || slug) +
          "</span>" +
          (have ? '<span class="ollama-chip is-on">Yours</span>' : "") +
          "</div>" +
          (row.description ? '<p class="general-tab-section-desc">' + esc(row.description) + "</p>" : "") +
          '<div class="ollama-tr-caps">' +
          (row.capabilities || [])
            .map(function (c) {
              return '<span class="ollama-chip is-on" data-no-translate>' + esc(c) + "</span>";
            })
            .join("") +
          (row.sizes || [])
            .map(function (s) {
              return '<span class="ollama-chip" data-no-translate>' + esc(s) + "</span>";
            })
            .join("") +
          '</div><div class="ollama-tr-meta">' +
          (row.pulls ? "<span>" + esc(row.pulls) + " pulls</span>" : "") +
          (row.tag_count ? "<span>" + esc(row.tag_count) + " tags</span>" : "") +
          (row.updated ? "<span>" + esc(row.updated) + "</span>" : "") +
          '</div></div><button type="button" class="settings-btn" data-act="pull" data-name="' +
          esc(slug) +
          '"' +
          (have ? " disabled" : "") +
          ">" +
          (have ? "Installed" : "Download") +
          "</button></div>"
        );
      })
      .join("");
    return (
      '<div class="ollama-search">' +
      icon("search") +
      '<input class="settings-input" data-act="query" type="search" placeholder="Search models…" value="' +
      esc(state.query) +
      '" /><button type="button" class="settings-btn" data-act="search"' +
      (state.busy ? " disabled" : "") +
      ">" +
      (state.busy ? "…" : "Search") +
      "</button></div>" +
      '<div class="ollama-filters">' +
      chips +
      '<button type="button" class="ollama-chip' +
      (state.order === "popular" ? " is-on" : "") +
      '" data-act="order" data-name="popular">Popular</button>' +
      '<button type="button" class="ollama-chip' +
      (state.order === "newest" ? " is-on" : "") +
      '" data-act="order" data-name="newest">Newest</button></div>' +
      (state.error ? '<p class="ollama-error" data-no-translate>' + esc(state.error) + "</p>" : "") +
      '<div class="ollama-table">' +
      (cards || (!state.busy ? '<p class="general-tab-section-desc">No models matched those filters.</p>' : "")) +
      "</div>"
    );
  }

  function boardHtml() {
    return (
      '<div class="ollama-board" data-no-translate><div class="ollama-board-head">' +
      icon("board") +
      "<h3>Ollama</h3>" +
      '<button type="button" class="settings-btn' +
      (state.tab === "yours" ? " is-on" : "") +
      '" data-act="tab" data-name="yours">Yours</button>' +
      '<button type="button" class="settings-btn' +
      (state.tab === "library" ? " is-on" : "") +
      '" data-act="tab" data-name="library">Library</button>' +
      '<button type="button" class="settings-btn' +
      (state.tab === "stats" ? " is-on" : "") +
      '" data-act="tab" data-name="stats">Stats</button></div>' +
      '<div class="ollama-board-main">' +
      (state.tab === "library" ? libraryPane() : state.tab === "stats" ? livePane() : yoursPane()) +
      "</div></div>"
    );
  }

  function renderMount() {
    var mount = document.getElementById(MOUNT_ID);
    if (!mount) return;
    mount.innerHTML = boardHtml();
    startLivePoll();
  }

  function emitPullJob(pull) {
    if (!pull) return;
    var job = {
      id: "ollama-pull:" + (pull.job_id || pull.model || "job"),
      source: "ollama",
      title: "Download " + (pull.model || "model"),
      detail: pull.error || pull.status || (pull.mode === "terminal" ? "Terminal pull" : "Ollama API"),
      percent: pull.percent,
      phase: pull.error ? "error" : pull.done ? "done" : "working",
      cancelable: !pull.done && !pull.error,
    };
    var host = window.__duckyPluginHost;
    if (host && host.jobs && typeof host.jobs.upsert === "function") {
      host.jobs.upsert(job);
      return;
    }
    window.dispatchEvent(new CustomEvent("ducky:background-job", { detail: job }));
  }

  function renderPicker() {
    if (!pickerMount) return;
    var st = state.settings;
    var name = state.selected || "";
    if (!name) {
      pickerMount.innerHTML = "";
      return;
    }
    if (!st || !st.ok) {
      pickerMount.innerHTML = '<p class="general-tab-section-desc">Loading sliders…</p>';
      return;
    }
    pickerMount.innerHTML =
      '<div class="ollama-picker" data-no-translate><div class="ollama-search"><span data-no-translate>' +
      esc(name) +
      '</span></div><div class="ollama-faders">' +
      ctxFaders(st) +
      optFaders(st, 3) +
      "</div></div>";
  }

  function maxCtxFor(name) {
    var row = ((state.local && state.local.models) || []).filter(function (m) {
      return m.name === name;
    })[0];
    return (row && row.context_length) || 0;
  }

  function attachPicker(mount, model) {
    ensureStyle();
    pickerMount = mount;
    var name = bareModel(model);
    if (name) state.selected = name;
    var go = function () {
      if (!state.selected) {
        renderPicker();
        return;
      }
      loadSettings(state.selected, maxCtxFor(state.selected));
    };
    if (state.local) go();
    else {
      call("local.list").then(function (res) {
        state.local = res;
        go();
      });
    }
  }

  function detachPicker() {
    if (pickerMount) pickerMount.innerHTML = "";
    pickerMount = null;
  }

  function attachSettings(mount) {
    ensureStyle();
    var existing = document.getElementById(MOUNT_ID);
    if (existing && existing !== mount) {
      existing.removeAttribute("id");
      existing.innerHTML = "";
    }
    mount.id = MOUNT_ID;
    state.boardOpen = true;
    render();
    if (!state.local) refreshLocal();
    refreshLive();
    refreshHistory();
  }

  function detachSettings() {
    var el = document.getElementById(MOUNT_ID);
    if (el) {
      el.innerHTML = "";
      el.removeAttribute("id");
    }
    state.boardOpen = false;
    if (localRetry) {
      clearTimeout(localRetry);
      localRetry = 0;
    }
    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = 0;
    }
  }

  function onLlmSlot(ev) {
    var d = (ev && ev.detail) || {};
    if (d.surface === "picker") {
      if (!isOllamaDetail(d) || !d.open || !d.mount) {
        detachPicker();
        return;
      }
      attachPicker(d.mount, d.model);
      return;
    }
    if (d.surface === "settings") {
      if (!isOllamaDetail(d) || !d.open || !d.mount) {
        detachSettings();
        return;
      }
      attachSettings(d.mount);
    }
  }

  function render() {
    renderMount();
    renderPicker();
  }

  function refreshLive() {
    return call("stats.live", {}).then(function (res) {
      if (res && res.ok && res.vram_total) {
        res.vram_pct = (100 * (res.vram_used || 0)) / res.vram_total;
      }
      state.live = res;
      if (res && res.ok) {
        state.samples = (state.samples || []).concat([res]).slice(-36);
      }
      if (state.boardOpen || (res && res.thinking)) render();
    });
  }

  function refreshHistory() {
    return call("history.list", {}).then(function (res) {
      state.history = (res && res.calls) || [];
      if (state.histIndex > state.history.length - 1) state.histIndex = Math.max(0, state.history.length - 1);
      if (state.boardOpen) render();
    });
  }

  function refreshLocal() {
    if (localRetry) {
      clearTimeout(localRetry);
      localRetry = 0;
    }
    return call("local.list", {}).then(function (res) {
      state.local = res;
      render();
      if (state.boardOpen && res && res.ok === false) {
        localRetry = setTimeout(refreshLocal, 2000);
      }
    });
  }

  function refreshLibrary() {
    state.busy = true;
    state.error = "";
    render();
    return call("library.search", {
      query: state.query,
      capabilities: state.caps,
      order: state.order,
    }).then(function (res) {
      state.busy = false;
      if (!res || !res.ok) state.error = (res && res.error) || "Could not load the Ollama library.";
      state.catalog = (res && res.models) || [];
      render();
    });
  }

  function loadSettings(name, maxCtx) {
    return call("settings.get", { model: name, max_ctx: maxCtx || 0 }).then(function (res) {
      state.settings = res;
      render();
    });
  }

  function startPullPoll() {
    if (pullTimer) clearInterval(pullTimer);
    if (!state.pull || !state.pull.job_id || state.pull.done) return;
    pullTimer = setInterval(function () {
      if (!state.pull || !state.pull.job_id) return;
      call("pull.status", { job_id: state.pull.job_id }).then(function (st) {
        state.pull = st;
        emitPullJob(st);
        if (state.boardOpen) render();
        if (st && st.done) {
          clearInterval(pullTimer);
          pullTimer = 0;
          if (!st.error) refreshLocal();
        }
      });
    }, 1200);
  }

  function startLivePoll() {
    if (!state.boardOpen || state.tab !== "stats") {
      if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = 0;
      }
      return;
    }
    if (liveTimer) return;
    liveTimer = setInterval(function () {
      if (!state.boardOpen || state.tab !== "stats") return;
      refreshLive();
      refreshHistory();
    }, 2000);
  }

  function onMountClick(ev) {
    var t = ev.target.closest("[data-act]");
    if (!t || !inOllamaUi(t)) return;
    var act = t.getAttribute("data-act");
    var name = t.getAttribute("data-name") || "";
    if (act === "tab") {
      state.tab = name;
      render();
      if (name === "library" && !state.catalog.length) refreshLibrary();
      if (name === "stats") {
        refreshLive();
        refreshHistory();
        startLivePoll();
      } else if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = 0;
      }
      return;
    }
    if (act === "select") {
      state.selected = state.selected === name ? "" : name;
      state.settings = null;
      render();
      if (state.selected) loadSettings(name, maxCtxFor(name));
      return;
    }
    if (act === "delete") {
      state.deleteName = name;
      render();
      return;
    }
    if (act === "delete-no") {
      state.deleteName = "";
      render();
      return;
    }
    if (act === "delete-yes") {
      var doomed = name || state.deleteName;
      call("local.delete", { name: doomed }).then(function (res) {
        if (res && res.ok) {
          if (state.selected === doomed) state.selected = "";
          state.deleteName = "";
          refreshLocal();
        } else {
          state.local = state.local || { ok: false };
          state.local.error = (res && res.error) || "Delete failed";
          render();
        }
      });
      return;
    }
    if (act === "pull") {
      call("pull.start", { name: name }).then(function (res) {
        state.pull = res;
        emitPullJob(res);
        startPullPoll();
      });
      return;
    }
    if (act === "search") {
      refreshLibrary();
      return;
    }
    if (act === "cap") {
      state.caps =
        state.caps.indexOf(name) >= 0
          ? state.caps.filter(function (c) {
              return c !== name;
            })
          : state.caps.concat([name]);
      refreshLibrary();
      return;
    }
    if (act === "order") {
      state.order = name;
      refreshLibrary();
      return;
    }
    if (act === "alive") {
      call("settings.set", {
        model: state.selected,
        keep_alive: Number(t.getAttribute("data-value")),
        max_ctx: maxCtxFor(state.selected),
      }).then(function (res) {
        state.settings = res;
        render();
      });
      return;
    }
    if (act === "reset") {
      var ticks = (state.settings && state.settings.ticks) || [];
      var maxTick = ticks[ticks.length - 1] || 0;
      call("settings.set", {
        model: state.selected,
        num_ctx: maxTick,
        keep_alive: -1,
        max_ctx: maxTick,
      }).then(function (res) {
        state.settings = res;
        render();
      });
      return;
    }
    if (act === "switch") {
      var cur = ((state.settings && state.settings.options) || {})[t.getAttribute("data-opt")];
      var bag = {};
      bag[t.getAttribute("data-opt")] = !cur;
      call("settings.set", { model: state.selected, options: bag }).then(function (res) {
        state.settings = res;
        render();
      });
      return;
    }
    if (act === "apply-hist") {
      call("history.apply", { index: Number(t.getAttribute("data-index") || 0), model: state.selected }).then(
        function (res) {
          state.settings = res;
          if (state.selected) render();
          refreshLocal();
        }
      );
    }
  }

  function onMountInput(ev) {
    var t = ev.target;
    if (!t.getAttribute) return;
    if (t.getAttribute("data-act") === "query") {
      state.query = t.value;
      return;
    }
    if (t.getAttribute("data-act") === "hist") {
      state.histIndex = Number(t.value || 0);
      render();
      return;
    }
    if (t.getAttribute("data-act") === "opt") {
      var raw = Number(t.value);
      var optId = t.getAttribute("data-opt");
      var next = {};
      next[optId] = raw;
      if (state.settings && state.settings.options) state.settings.options[optId] = raw;
      var fader = t.closest(".ollama-fader");
      if (fader) {
        var heat = heatOf(raw, t.min, t.max);
        fader.style.setProperty("--effort-heat", String(heat));
        fader.style.setProperty("--slider-color", heatColor(heat));
        var valEl = fader.querySelector(".ollama-fader-val");
        if (valEl) valEl.textContent = String(raw);
      }
      call("settings.set", { model: state.selected, options: next }).then(function (res) {
        if (res && res.ok) state.settings = res;
      });
      return;
    }
    if (t.getAttribute("data-act") === "ctx") {
      var ticks = (state.settings && state.settings.ticks) || [];
      var nxt = ticks[Number(t.value)];
      if (!nxt || !state.selected) return;
      call("settings.set", {
        model: state.selected,
        num_ctx: nxt,
        max_ctx: ticks[ticks.length - 1] || nxt,
      }).then(function (res) {
        state.settings = res;
        render();
      });
    }
  }

  function onJobAction(ev) {
    var d = (ev && ev.detail) || {};
    if (d.action !== "cancel" || !state.pull || !state.pull.job_id) return;
    var want = "ollama-pull:" + state.pull.job_id;
    if (d.id !== want && d.id !== "ollama-pull:" + (state.pull.model || "")) return;
    call("pull.cancel", { job_id: state.pull.job_id });
    emitPullJob({
      job_id: state.pull.job_id,
      model: state.pull.model,
      status: "cancelled",
      done: true,
      error: "Cancelled",
      percent: state.pull.percent,
    });
    state.pull = null;
    if (pullTimer) {
      clearInterval(pullTimer);
      pullTimer = 0;
    }
  }

  function trySlotsOnce() {
    var settings = ollamaSlide();
    if (settings) attachSettings(settings);
    var picker = document.querySelector('[data-ducky-llm-slot="picker"][data-provider="ollama"]');
    if (picker) attachPicker(picker, picker.getAttribute("data-model") || "");
  }

  document.addEventListener("click", onMountClick);
  document.addEventListener("input", onMountInput);
  window.addEventListener("ducky:llm-slot", onLlmSlot);
  window.addEventListener("ducky:background-job-action", onJobAction);
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && ev.target && ev.target.getAttribute && ev.target.getAttribute("data-act") === "query") {
      ev.preventDefault();
      refreshLibrary();
    }
    if (ev.key === "Escape" && state.deleteName) {
      state.deleteName = "";
      render();
    }
  });

  trySlotsOnce();

  window.__duckyPluginBootCleanups = window.__duckyPluginBootCleanups || {};
  window.__duckyPluginBootCleanups[PLUGIN_ID] = function () {
    window.removeEventListener("ducky:llm-slot", onLlmSlot);
    window.removeEventListener("ducky:background-job-action", onJobAction);
    if (pullTimer) clearInterval(pullTimer);
    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = 0;
    }
    detachPicker();
    detachSettings();
    document.removeEventListener("click", onMountClick);
    document.removeEventListener("input", onMountInput);
    var style = document.getElementById(STYLE_ID);
    if (style) style.remove();
  };
})();
