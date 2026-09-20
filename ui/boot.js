/**
 * Inject the Ollama library/Yours manager into Settings → LLMs → Ollama.
 * Main-window boot so we use host CSS variables. No EXE changes.
 */
(function () {
  "use strict";

  var PLUGIN_ID = "ollama";
  var MOUNT_ID = "ollama-library-mount";
  var STYLE_ID = "ollama-library-style";
  var MODAL_ID = "ollama-library-modal";
  var CAPS = ["vision", "tools", "thinking", "embedding", "cloud"];
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
  };
  var observer = null;
  var pullTimer = 0;
  var liveTimer = 0;

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

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var el = document.createElement("style");
    el.id = STYLE_ID;
    el.textContent = [
      "#ollama-library-mount{margin-top:8px}",
      ".ollama-lib-tabs{display:flex;gap:8px;margin:0 0 12px}",
      ".ollama-lib-tabs .settings-btn.is-on{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 18%,transparent)}",
      ".ollama-lib-card{display:flex;flex-direction:column;gap:10px}",
      ".ollama-lib-storage{display:flex;align-items:baseline;gap:10px;color:var(--text)}",
      ".ollama-lib-muted{color:var(--muted);font-size:12px}",
      ".ollama-lib-search,.ollama-lib-filters,.ollama-lib-keep,.ollama-lib-row{display:flex;align-items:center;gap:8px}",
      ".ollama-lib-search .settings-input{flex:1}",
      ".ollama-lib-filters{flex-wrap:wrap}",
      ".ollama-lib-row{align-items:flex-start;padding:8px 0;border-top:1px solid var(--border)}",
      ".ollama-lib-row.is-selected{background:color-mix(in srgb,var(--accent) 10%,transparent);margin:0 -10px;padding:8px 10px;border-radius:8px}",
      ".ollama-lib-row-main{flex:1;min-width:0;text-align:left;background:none;border:0;padding:0;color:inherit;cursor:pointer}",
      ".ollama-lib-row-title{display:flex;align-items:center;gap:8px;font-weight:600;color:var(--text)}",
      ".ollama-lib-meta,.ollama-lib-chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}",
      ".ollama-lib-meta{color:var(--muted);font-size:12px}",
      ".ollama-lib-chip,.ollama-lib-chip-btn{border:1px solid var(--border);background:var(--hover);color:var(--muted);border-radius:999px;padding:2px 8px;font-size:11px;line-height:1.4}",
      ".ollama-lib-chip-btn{cursor:pointer;font:inherit}",
      ".ollama-lib-chip.is-on,.ollama-lib-chip-btn.is-on{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--text)}",
      ".ollama-lib-tune{border-top:1px solid var(--border);padding-top:12px}",
      ".ollama-lib-tune-head{display:flex;justify-content:space-between;gap:8px;margin-bottom:8px}",
      ".ollama-lib-slider{width:100%;accent-color:var(--accent)}",
      ".ollama-lib-ticks{display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin:4px 0 12px}",
      "#ollama-library-modal{position:fixed;inset:0;z-index:100040;display:flex;align-items:center;justify-content:center;background:color-mix(in srgb,var(--bg) 55%,transparent)}",
      "#ollama-library-modal .ollama-lib-dialog{width:min(480px,calc(100vw - 32px));background:var(--card);border:1px solid var(--border);border-radius:var(--radius,12px);padding:20px;color:var(--text)}",
      ".ollama-lib-progress-track{height:6px;border-radius:999px;background:var(--hover);overflow:hidden;margin:8px 0}",
      ".ollama-lib-progress-track i{display:block;height:100%;background:var(--accent)}",
      ".ollama-lib-actions{display:flex;justify-content:flex-end;gap:8px;margin-top:16px}",
      ".ollama-lib-bar{height:8px;border-radius:999px;background:var(--hover);overflow:hidden;margin:4px 0 10px}",
      ".ollama-lib-bar i{display:block;height:100%;background:var(--accent)}",
      ".ollama-lib-bar.is-hot i{background:var(--amber)}",
      ".ollama-lib-bar.is-max i{background:var(--red)}",
      ".ollama-lib-opt{margin:8px 0}",
      ".ollama-lib-svg{width:100%;height:56px;display:block}",
    ].join("");
    document.head.appendChild(el);
  }

  function ollamaSlide() {
    var title = document.querySelector(".duckies-tab-detail-title");
    if (!title || String(title.textContent || "").trim() !== "Ollama") return null;
    return document.querySelector(".duckies-tab-detail-scroll");
  }

  function chip(label, on) {
    return '<span class="ollama-lib-chip' + (on ? " is-on" : "") + '">' + esc(label) + "</span>";
  }

  function barHtml(pct, extra) {
    var n = Math.max(0, Math.min(100, Number(pct) || 0));
    var cls = n >= 90 ? " is-max" : n >= 70 ? " is-hot" : "";
    return (
      '<div class="ollama-lib-bar' +
      cls +
      (extra ? " " + extra : "") +
      '"><i style="width:' +
      n +
      '%"></i></div>'
    );
  }

  function sparkline(values, colorVar) {
    var pts = values || [];
    if (!pts.length) return '<svg class="ollama-lib-svg" viewBox="0 0 100 40"></svg>';
    var max = Math.max.apply(null, pts.concat([1]));
    var w = 100;
    var step = pts.length > 1 ? w / (pts.length - 1) : w;
    var d = pts
      .map(function (v, i) {
        var x = (i * step).toFixed(1);
        var y = (40 - (Number(v) / max) * 36 - 2).toFixed(1);
        return x + "," + y;
      })
      .join(" ");
    return (
      '<svg class="ollama-lib-svg" viewBox="0 0 100 40" preserveAspectRatio="none">' +
      '<polyline fill="none" stroke="var(' +
      (colorVar || "--accent") +
      ')" stroke-width="1.6" points="' +
      d +
      '" /></svg>'
    );
  }

  function optSlidersHtml(st) {
    var opts = (st && st.options) || {};
    var sliders = (st && st.sliders) || [];
    var switches = (st && st.switches) || [];
    var html = sliders
      .map(function (s) {
        var val = opts[s.id] != null ? opts[s.id] : s.default;
        var lo = s.lo;
        var hi = s.hi;
        var step = s.step;
        return (
          '<div class="ollama-lib-opt"><div class="ollama-lib-tune-head"><span>' +
          esc(s.label) +
          '</span><span class="ollama-lib-muted">' +
          val +
          "</span></div>" +
          '<input type="range" class="ollama-lib-slider" data-act="opt" data-opt="' +
          esc(s.id) +
          '" min="' +
          lo +
          '" max="' +
          hi +
          '" step="' +
          step +
          '" value="' +
          val +
          '" /></div>'
        );
      })
      .join("");
    var sw = switches
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
      .join("");
    return html + (sw ? '<div class="ollama-lib-keep">' + sw + "</div>" : "");
  }

  function liveHtml() {
    var live = state.live || {};
    var samples = state.samples || [];
    var cpuPts = samples.map(function (s) {
      return s.cpu_pct || 0;
    });
    var ramPts = samples.map(function (s) {
      return s.ram_pct || 0;
    });
    var gpuPts = samples.map(function (s) {
      return s.gpu_pct || 0;
    });
    var calls = state.history || [];
    var idx = Math.max(0, Math.min(calls.length - 1, state.histIndex));
    var call = calls[idx];
    var runNote = live.thinking
      ? "Thinking · " + (live.generating_model || "")
      : live.running
        ? "Loaded"
        : "Idle";
    var hist = "";
    if (calls.length) {
      hist =
        '<div class="ollama-lib-tune"><div class="ollama-lib-tune-head"><span>Call history</span>' +
        '<span class="ollama-lib-muted">' +
        (idx + 1) +
        " / " +
        calls.length +
        "</span></div>" +
        '<input type="range" class="ollama-lib-slider" data-act="hist" min="0" max="' +
        (calls.length - 1) +
        '" value="' +
        idx +
        '" />' +
        '<p class="general-tab-section-desc">' +
        esc(call.model || "") +
        " · " +
        (call.tokens_per_s || 0) +
        " tok/s · " +
        (call.elapsed_s || 0) +
        "s · ctx " +
        (call.num_ctx || "?") +
        " · " +
        (call.completion_tokens || 0) +
        " out</p>" +
        sparkline(
          calls.map(function (c) {
            return c.tokens_per_s || 0;
          }),
          "--green"
        ) +
        '<button type="button" class="settings-btn" data-act="apply-hist" data-index="' +
        idx +
        '">Apply these sliders</button></div>';
    } else {
      hist = '<p class="general-tab-section-desc">No calls logged yet. Chat once and the last 5 runs show up here.</p>';
    }
    var models = (live.models || [])
      .map(function (m) {
        return (
          '<div class="ollama-lib-meta"><span>' +
          esc(m.name) +
          "</span><span>" +
          esc(m.size_vram_label || "") +
          " VRAM</span></div>"
        );
      })
      .join("");
    return (
      '<div class="llms-provider-card ollama-lib-card">' +
      '<div class="ollama-lib-storage"><span>' +
      runNote +
      "</span><strong>" +
      (live.thinking ? "live" : live.running ? "warm" : "idle") +
      "</strong></div>" +
      '<div class="ollama-lib-tune-head"><span>CPU</span><span class="ollama-lib-muted">' +
      (live.cpu_pct != null ? live.cpu_pct + "%" : "n/a") +
      "</span></div>" +
      barHtml(live.cpu_pct) +
      sparkline(cpuPts, "--accent") +
      '<div class="ollama-lib-tune-head"><span>RAM</span><span class="ollama-lib-muted">' +
      esc(live.ram_label || "") +
      "</span></div>" +
      barHtml(live.ram_pct) +
      sparkline(ramPts, "--amber") +
      (live.vram_label
        ? '<div class="ollama-lib-tune-head"><span>GPU VRAM</span><span class="ollama-lib-muted">' +
          esc(live.vram_label) +
          (live.gpu_pct != null ? " · " + live.gpu_pct + "%" : "") +
          "</span></div>" +
          barHtml(live.gpu_pct != null ? live.gpu_pct : live.vram_total ? (100 * live.vram_used) / live.vram_total : 0) +
          sparkline(gpuPts, "--green")
        : "") +
      models +
      hist +
      "</div>"
    );
  }

  function yoursHtml() {
    var local = state.local || {};
    var models = local.models || [];
    var rows = models
      .map(function (m) {
        var caps = (m.capabilities || []).map(function (c) {
          return String(c).toLowerCase();
        });
        var sel = m.name === state.selected ? " is-selected" : "";
        return (
          '<div class="ollama-lib-row' +
          sel +
          '">' +
          '<button type="button" class="ollama-lib-row-main" data-act="select" data-name="' +
          esc(m.name) +
          '">' +
          '<div class="ollama-lib-row-title"><span>' +
          esc(m.name) +
          "</span>" +
          (m.loaded ? chip("Loaded", true) : "") +
          "</div>" +
          '<div class="ollama-lib-meta"><span>' +
          esc(m.size_label || "") +
          "</span>" +
          (m.parameter_size ? "<span>" + esc(m.parameter_size) + "</span>" : "") +
          (m.quantization ? "<span>" + esc(m.quantization) + "</span>" : "") +
          (m.context_length ? "<span>" + Math.round(m.context_length / 1024) + "k ctx</span>" : "") +
          (m.loaded && m.size_vram_label ? "<span>" + esc(m.size_vram_label) + " VRAM</span>" : "") +
          "</div>" +
          '<div class="ollama-lib-chips">' +
          ["vision", "tools", "thinking"]
            .filter(function (c) {
              return caps.indexOf(c) >= 0;
            })
            .map(function (c) {
              return chip(c, true);
            })
            .join("") +
          "</div></button>" +
          '<button type="button" class="settings-btn" data-act="delete" data-name="' +
          esc(m.name) +
          '">Delete</button></div>'
        );
      })
      .join("");
    var tune = "";
    var st = state.settings;
    if (state.selected && st && st.ok) {
      var ticks = st.ticks || [];
      var labels = st.tick_labels || [];
      var idx = Math.max(0, ticks.indexOf(st.num_ctx));
      if (ticks.indexOf(st.num_ctx) < 0) idx = Math.max(0, ticks.length - 1);
      tune =
        '<div class="ollama-lib-tune"><div class="ollama-lib-tune-head"><span>Context length</span>' +
        '<span class="ollama-lib-muted">' +
        esc(labels[idx] || st.num_ctx || "") +
        "</span></div>" +
        '<input type="range" class="ollama-lib-slider" data-act="ctx" min="0" max="' +
        Math.max(0, ticks.length - 1) +
        '" value="' +
        idx +
        '" />' +
        '<div class="ollama-lib-ticks">' +
        labels
          .map(function (l) {
            return "<span>" + esc(l) + "</span>";
          })
          .join("") +
        "</div>" +
        '<div class="ollama-lib-keep">' +
        '<button type="button" class="settings-btn' +
        (st.keep_alive === -1 ? " is-on" : "") +
        '" data-act="alive" data-value="-1">Stay loaded</button>' +
        '<button type="button" class="settings-btn' +
        (st.keep_alive === 900 ? " is-on" : "") +
        '" data-act="alive" data-value="900">15m</button>' +
        '<button type="button" class="settings-btn' +
        (st.keep_alive === 0 ? " is-on" : "") +
        '" data-act="alive" data-value="0">Unload idle</button>' +
        '<button type="button" class="settings-btn" data-act="reset">Reset to defaults</button></div></div>' +
        optSlidersHtml(st);
    }
    var strip = "";
    var live = state.live || {};
    if (live.thinking || live.running) {
      strip =
        '<div class="ollama-lib-tune"><div class="ollama-lib-storage"><span>' +
        (live.thinking ? "Thinking · " + esc(live.generating_model || "") : "Loaded") +
        '</span><strong>' +
        (live.cpu_pct != null ? live.cpu_pct + "% CPU" : "live") +
        "</strong></div>" +
        barHtml(live.cpu_pct) +
        barHtml(live.ram_pct) +
        "</div>";
    }
    return (
      '<div class="llms-provider-card ollama-lib-card">' +
      strip +
      '<div class="ollama-lib-storage"><span>Storage</span><strong>' +
      esc(local.storage_label || "0 B") +
      "</strong><span class=\"ollama-lib-muted\">" +
      (local.count || 0) +
      " model" +
      ((local.count || 0) === 1 ? "" : "s") +
      (local.loaded_count ? " · " + local.loaded_count + " loaded" : "") +
      "</span></div>" +
      (local.error ? '<p class="llms-provider-status-text is-fail">' + esc(local.error) + "</p>" : "") +
      (models.length ? rows : '<p class="general-tab-section-desc">Nothing pulled yet. Open Library to download a model.</p>') +
      tune +
      "</div>"
    );
  }

  function libraryHtml() {
    var installed = {};
    ((state.local && state.local.models) || []).forEach(function (m) {
      installed[String(m.name || "").split(":")[0]] = true;
    });
    var cards = (state.catalog || [])
      .map(function (row) {
        var slug = row.slug || "";
        var have = installed[slug.split("/").pop()];
        return (
          '<div class="ollama-lib-row"><div class="ollama-lib-row-main">' +
          '<div class="ollama-lib-row-title"><span>' +
          esc(row.name || slug) +
          "</span>" +
          (have ? chip("Yours", true) : "") +
          "</div>" +
          (row.description ? '<p class="general-tab-section-desc">' + esc(row.description) + "</p>" : "") +
          '<div class="ollama-lib-chips">' +
          (row.capabilities || [])
            .map(function (c) {
              return chip(c, true);
            })
            .join("") +
          (row.sizes || [])
            .map(function (s) {
              return chip(s, false);
            })
            .join("") +
          "</div>" +
          '<div class="ollama-lib-meta">' +
          (row.pulls ? "<span>" + esc(row.pulls) + " Pulls</span>" : "") +
          (row.tag_count ? "<span>" + esc(row.tag_count) + " Tags</span>" : "") +
          (row.updated ? "<span>Updated " + esc(row.updated) + "</span>" : "") +
          "</div></div>" +
          '<button type="button" class="settings-btn" data-act="pull" data-name="' +
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
      '<div class="llms-provider-card ollama-lib-card">' +
      '<div class="ollama-lib-search"><input class="settings-input" data-act="query" type="search" placeholder="Search models…" value="' +
      esc(state.query) +
      '" />' +
      '<button type="button" class="settings-btn" data-act="search"' +
      (state.busy ? " disabled" : "") +
      ">" +
      (state.busy ? "Loading…" : "Search") +
      "</button></div>" +
      '<div class="ollama-lib-filters">' +
      CAPS.map(function (id) {
        return (
          '<button type="button" class="ollama-lib-chip-btn' +
          (state.caps.indexOf(id) >= 0 ? " is-on" : "") +
          '" data-act="cap" data-name="' +
          id +
          '">' +
          id +
          "</button>"
        );
      }).join("") +
      '<button type="button" class="settings-btn' +
      (state.order === "popular" ? " is-on" : "") +
      '" data-act="order" data-name="popular">Popular</button>' +
      '<button type="button" class="settings-btn' +
      (state.order === "newest" ? " is-on" : "") +
      '" data-act="order" data-name="newest">Newest</button></div>' +
      (state.error ? '<p class="llms-provider-status-text is-fail">' + esc(state.error) + "</p>" : "") +
      (cards ||
        (!state.busy ? '<p class="general-tab-section-desc">No models matched those filters.</p>' : "")) +
      "</div>"
    );
  }

  function renderMount() {
    var mount = document.getElementById(MOUNT_ID);
    if (!mount) return;
    mount.innerHTML =
      '<h3 class="general-tab-section-title" style="margin:0 0 6px">Models</h3>' +
      '<p class="general-tab-section-desc">Browse the real Ollama library, download into this server, and manage disk from Yours.</p>' +
      '<div class="ollama-lib-tabs">' +
      '<button type="button" class="settings-btn' +
      (state.tab === "yours" ? " is-on" : "") +
      '" data-act="tab" data-name="yours">Yours</button>' +
      '<button type="button" class="settings-btn' +
      (state.tab === "library" ? " is-on" : "") +
      '" data-act="tab" data-name="library">Library</button>' +
      '<button type="button" class="settings-btn' +
      (state.tab === "live" ? " is-on" : "") +
      '" data-act="tab" data-name="live">Live</button></div>' +
      (state.tab === "yours" ? yoursHtml() : state.tab === "library" ? libraryHtml() : liveHtml());
  }

  function renderModal() {
    var old = document.getElementById(MODAL_ID);
    var pull = state.pull;
    var del = state.deleteName;
    if (!pull && !del) {
      if (old) old.remove();
      return;
    }
    var box = old || document.createElement("div");
    box.id = MODAL_ID;
    if (pull) {
      var pct = pull.percent == null ? 0 : Math.max(0, Math.min(100, pull.percent));
      box.innerHTML =
        '<div class="ollama-lib-dialog"><h3 style="margin:0 0 8px">Download ' +
        esc(pull.model || "") +
        "</h3>" +
        '<p class="general-tab-section-desc">' +
        (pull.mode === "terminal"
          ? "The ollama pull tab in the terminal list is doing the download. This modal shows the same progress."
          : "Pulling through your Ollama server API. Progress updates here.") +
        "</p>" +
        '<div class="ollama-lib-progress-track"><i style="width:' +
        pct +
        '%"></i></div>' +
        '<span class="ollama-lib-muted">' +
        (pull.percent != null ? Math.round(pull.percent) + "%" : "…") +
        " · " +
        esc(pull.status || "starting") +
        "</span>" +
        (pull.error ? '<p class="llms-provider-status-text is-fail">' + esc(pull.error) + "</p>" : "") +
        '<div class="ollama-lib-actions"><button type="button" class="settings-btn" data-act="pull-close">' +
        (pull.done ? "Done" : "Cancel") +
        "</button></div></div>";
    } else {
      box.innerHTML =
        '<div class="ollama-lib-dialog"><h3 style="margin:0 0 8px">Delete model</h3>' +
        '<p class="general-tab-section-desc">Remove <strong>' +
        esc(del) +
        "</strong> from this Ollama server. This frees the disk used by that model.</p>" +
        '<div class="ollama-lib-actions">' +
        '<button type="button" class="settings-btn" data-act="delete-no">Cancel</button>' +
        '<button type="button" class="settings-btn" data-act="delete-yes">Delete</button></div></div>';
    }
    if (!old) document.body.appendChild(box);
  }

  function render() {
    renderMount();
    renderModal();
  }

  function refreshLive() {
    return call("stats.live", {}).then(function (res) {
      state.live = res;
      if (res && res.ok) {
        state.samples = (state.samples || []).concat([res]).slice(-30);
      }
      if (state.tab === "live" || (res && res.thinking)) render();
    });
  }

  function refreshHistory() {
    return call("history.list", {}).then(function (res) {
      state.history = (res && res.calls) || [];
      if (state.histIndex > state.history.length - 1) state.histIndex = Math.max(0, state.history.length - 1);
      if (state.tab === "live") render();
    });
  }

  function refreshLocal() {
    return call("local.list", {}).then(function (res) {
      state.local = res;
      render();
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
        render();
        if (st && st.done) {
          clearInterval(pullTimer);
          pullTimer = 0;
          if (!st.error) refreshLocal();
        }
      });
    }, 1200);
  }

  function onMountClick(ev) {
    var t = ev.target.closest("[data-act]");
    if (!t || !document.getElementById(MOUNT_ID) || !document.getElementById(MOUNT_ID).contains(t)) return;
    var act = t.getAttribute("data-act");
    var name = t.getAttribute("data-name") || "";
    if (act === "tab") {
      state.tab = name;
      render();
      if (name === "library" && !state.catalog.length) refreshLibrary();
      if (name === "live") {
        refreshLive();
        refreshHistory();
      }
      startLivePoll();
      return;
    }
    if (act === "select") {
      state.selected = state.selected === name ? "" : name;
      state.settings = null;
      render();
      if (state.selected) {
        var row = ((state.local && state.local.models) || []).filter(function (m) {
          return m.name === name;
        })[0];
        loadSettings(name, row && row.context_length);
      }
      return;
    }
    if (act === "delete") {
      state.deleteName = name;
      render();
      return;
    }
    if (act === "pull") {
      call("pull.start", { name: name }).then(function (res) {
        state.pull = res;
        render();
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
        max_ctx: ((state.local && state.local.models) || []).filter(function (m) {
          return m.name === state.selected;
        })[0]
          ? ((state.local && state.local.models) || []).filter(function (m) {
              return m.name === state.selected;
            })[0].context_length
          : 0,
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
      var opts = {};
      opts[t.getAttribute("data-opt")] = !cur;
      call("settings.set", { model: state.selected, options: opts }).then(function (res) {
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
      var bag = {};
      var raw = Number(t.value);
      var optId = t.getAttribute("data-opt");
      bag[optId] = raw;
      if (state.settings && state.settings.options) state.settings.options[optId] = raw;
      var label = t.parentNode && t.parentNode.querySelector(".ollama-lib-muted");
      if (label) label.textContent = raw;
      call("settings.set", { model: state.selected, options: bag }).then(function (res) {
        if (res && res.ok) state.settings = res;
      });
      return;
    }
    if (t.getAttribute("data-act") === "ctx") {
      var ticks = (state.settings && state.settings.ticks) || [];
      var next = ticks[Number(t.value)];
      if (!next || !state.selected) return;
      call("settings.set", {
        model: state.selected,
        num_ctx: next,
        max_ctx: ticks[ticks.length - 1] || next,
      }).then(function (res) {
        state.settings = res;
        render();
      });
    }
  }

  function onModalClick(ev) {
    var t = ev.target.closest("[data-act]");
    if (!t) return;
    var box = document.getElementById(MODAL_ID);
    if (!box || !box.contains(t)) return;
    var act = t.getAttribute("data-act");
    if (act === "pull-close") {
      if (state.pull && state.pull.job_id && !state.pull.done) {
        call("pull.cancel", { job_id: state.pull.job_id });
      }
      state.pull = null;
      if (pullTimer) {
        clearInterval(pullTimer);
        pullTimer = 0;
      }
      render();
      return;
    }
    if (act === "delete-no") {
      state.deleteName = "";
      render();
      return;
    }
    if (act === "delete-yes") {
      var doomed = state.deleteName;
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
    }
  }

  function attachMount(scroll) {
    ensureStyle();
    var mount = document.getElementById(MOUNT_ID);
    if (mount && scroll.contains(mount)) {
      return;
    }
    if (mount) mount.remove();
    mount = document.createElement("section");
    mount.id = MOUNT_ID;
    mount.className = "general-tab-section";
    scroll.appendChild(mount);
    render();
    if (!state.local) refreshLocal();
    startLivePoll();
  }

  function startLivePoll() {
    if (state.tab !== "live") {
      if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = 0;
      }
      return;
    }
    if (liveTimer) return;
    liveTimer = setInterval(function () {
      if (!ollamaSlide() || state.tab !== "live") return;
      refreshLive();
      refreshHistory();
    }, 2000);
  }

  function tick() {
    var scroll = ollamaSlide();
    if (scroll) attachMount(scroll);
    else {
      var orphan = document.getElementById(MOUNT_ID);
      if (orphan) orphan.remove();
      if (liveTimer) {
        clearInterval(liveTimer);
        liveTimer = 0;
      }
    }
  }

  document.addEventListener("click", onMountClick);
  document.addEventListener("click", onModalClick);
  document.addEventListener("input", onMountInput);
  document.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter" && ev.target && ev.target.getAttribute && ev.target.getAttribute("data-act") === "query") {
      ev.preventDefault();
      refreshLibrary();
    }
  });

  observer = new MutationObserver(tick);
  observer.observe(document.body, { childList: true, subtree: true });
  tick();

  window.__duckyPluginBootCleanups = window.__duckyPluginBootCleanups || {};
  window.__duckyPluginBootCleanups[PLUGIN_ID] = function () {
    if (observer) observer.disconnect();
    if (pullTimer) clearInterval(pullTimer);
    if (liveTimer) {
      clearInterval(liveTimer);
      liveTimer = 0;
    }
    document.removeEventListener("click", onMountClick);
    document.removeEventListener("click", onModalClick);
    document.removeEventListener("input", onMountInput);
    var mount = document.getElementById(MOUNT_ID);
    if (mount) mount.remove();
    var modal = document.getElementById(MODAL_ID);
    if (modal) modal.remove();
    var style = document.getElementById(STYLE_ID);
    if (style) style.remove();
  };
})();
