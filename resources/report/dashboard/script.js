// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
document.querySelectorAll(".tab-link").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-link").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
  });
});

// ---------------------------------------------------------------------------
// Filter widgets (Choices.js)
// ---------------------------------------------------------------------------
const choicesInstances = {};

const choicesConfig = {
  searchEnabled: false,
  removeItemButton: true,
  shouldSort: false,
  itemSelectText: "",
  placeholder: false,
};

// Guard: region/season/init may be absent when stratification doesn't include them
function initChoices(id) {
  if (document.getElementById(id)) {
    choicesInstances[id] = new Choices("#" + id, choicesConfig);
    document.getElementById(id).addEventListener("change", scheduleUpdate);
  }
}

initChoices("region-select");
initChoices("season-select");
initChoices("init-select");
initChoices("source-select");
initChoices("metric-select");
initChoices("param-select");

function getSelected(id) {
  return choicesInstances[id] ? choicesInstances[id].getValue(true) : [];
}

// Initial chart
updateChart()


// ---- System metrics tab ----

const sysDataEl = document.getElementById("sysmetrics-data");
const sysData = sysDataEl ? JSON.parse(sysDataEl.textContent) : [];

if (sysData.length > 0) {
  choicesInstances["sys-model-type-select"] = new Choices("#sys-model-type-select", {
    searchEnabled: false,
    removeItemButton: true,
    shouldSort: false,
    itemSelectText: "",
    placeholder: false,
  });
  document.getElementById("sys-model-type-select").addEventListener("change", updateSysChart);

  choicesInstances["sys-source-select"] = new Choices("#sys-source-select", {
    searchEnabled: false,
    removeItemButton: true,
    shouldSort: false,
    itemSelectText: "",
    placeholder: false,
  });
  document.getElementById("sys-source-select").addEventListener("change", updateSysChart);

  // Populate metric filter from the data, then initialise Choices on it
  const sysMetricEl = document.getElementById("sys-metric-select");
  [...new Set(sysData.map(d => d.metric))].sort().forEach(m => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    opt.selected = true;
    sysMetricEl.appendChild(opt);
  });
  choicesInstances["sys-metric-select"] = new Choices("#sys-metric-select", {
    searchEnabled: false,
    removeItemButton: true,
    shouldSort: false,
    itemSelectText: "",
    placeholder: false,
  });
  document.getElementById("sys-metric-select").addEventListener("change", updateSysChart);

  const sysSpec = {
    "data": { "values": sysData },
    "facet": { "field": "metric", "type": "nominal", "title": null },
    "columns": 4,
    "resolve": { "scale": { "x": "shared", "y": "independent" } },
    "spec": {
      "params": [
        {
          "name": "xZoom",
          "select": {
            "type": "interval",
            "encodings": ["x"],
            "zoom": "wheel![!event.shiftKey]"
          },
          "bind": "scales"
        }
      ],
      "width": 300,
      "height": 200,
      "mark": { "type": "line", "point": { "filled": true, "size": 50 } },
      "encoding": {
        "x": {
          "field": "init_time",
          "type": "temporal",
          "title": null,
          "axis": { "labelAngle": -30, "format": "%b %d" }
        },
        "y": {
          "field": "value",
          "type": "quantitative",
          "title": null,
          "scale": { "zero": true }
        },
        "color": {
          "field": "source",
          "type": "nominal",
          "legend": { "orient": "top", "title": "Source" }
        },
        "shape": {
          "field": "model_type",
          "type": "nominal",
          "legend": { "orient": "top", "title": "Model type" }
        },
        "strokeDash": {
          "field": "model_type",
          "type": "nominal",
          "legend": { "orient": "top", "title": "Model type" }
        },
        "tooltip": [
          { "field": "source", "type": "nominal", "title": "Source" },
          { "field": "model_type", "type": "nominal", "title": "Model type" },
          { "field": "init_time", "type": "temporal", "title": "Init time", "format": "%Y-%m-%d %H:%M" },
          { "field": "metric", "type": "nominal", "title": "Metric" },
          { "field": "value", "type": "quantitative", "title": "Value", "format": ".3f" },
          { "field": "n_gpu", "type": "quantitative", "title": "GPUs" },
          { "field": "job_id", "type": "nominal", "title": "Job ID" },
        ],
      },
    },
  };

  function updateSysChart() {
    const selectedModelTypes = getSelected("sys-model-type-select");
    const selectedSources = getSelected("sys-source-select");
    const selectedMetrics = getSelected("sys-metric-select");
    const newSpec = JSON.parse(JSON.stringify(sysSpec));
    const filters = [];
    if (selectedModelTypes.length > 0) {
      filters.push({ field: "model_type", oneOf: selectedModelTypes });
    }
    if (selectedSources.length > 0) {
      filters.push({ field: "source", oneOf: selectedSources });
    }
    if (selectedMetrics.length > 0) {
      filters.push({ field: "metric", oneOf: selectedMetrics });
    }
    if (filters.length > 0) {
      newSpec.transform = [{ filter: { and: filters } }];
    }
    vegaEmbed("#sys-vis", newSpec, { actions: false });
  }

  updateSysChart();
}
// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
(function () {
  const raw = JSON.parse(document.getElementById("verif-data").textContent);
  // Convert columnar format {columns, data} → array of objects, and add derived column
  const cols = raw.columns;
  window.DATA = raw.data.map(row => {
    const obj = {};
    for (let i = 0; i < cols.length; i++) obj[cols[i]] = row[i];
    obj.region_season_init =
      "Region: " + obj.region + ", Season: " + obj.season + ", Init: " + obj.init_hour;
    return obj;
  });
})();
const DATA = window.DATA;

// ---------------------------------------------------------------------------
// Shared color scale — pin all sources to consistent colors across all cells
// ---------------------------------------------------------------------------
const ALL_SOURCES = [...new Set(DATA.map(d => d.source))];

// ---------------------------------------------------------------------------
// Vega view lifecycle
// ---------------------------------------------------------------------------
const vegaViews = new Map();   // key → Vega view

function disposeView(key) {
  if (vegaViews.has(key)) {
    try { vegaViews.get(key).finalize(); } catch (e) {}
    vegaViews.delete(key);
  }
}

function cellKey(metric, param) { return metric + "\x00" + param; }

// ---------------------------------------------------------------------------
// Cell spec
// ---------------------------------------------------------------------------
function makeCellSpec(cellData) {
  return {
    data: { values: cellData },
    config: { scale: { continuousPadding: 1 } },
    params: [{
      name: "xZoom",
      select: { type: "interval", encodings: ["x"], zoom: "wheel[!event.shiftKey]" },
      bind: "scales",
    }],
    transform: [{ filter: { param: "xZoom" } }],
    mark: { type: "line", point: { size: 40 } },
    width: 280,
    height: 160,
    encoding: {
      x: {
        field: "step",
        type: "quantitative",
        title: "Lead time (h)",
      },
      y: {
        field: "value",
        type: "quantitative",
        scale: { zero: false },
        title: null,
      },
      color: {
        field: "source",
        type: "nominal",
        scale: { domain: ALL_SOURCES },
        legend: null,
      },
      shape: {
        field: "region_season_init",
        type: "nominal",
        legend: null,
      },
      strokeDash: {
        field: "region_season_init",
        type: "nominal",
        legend: null,
      },
      tooltip: [
        { field: "source",            type: "nominal",      title: "Source" },
        { field: "region_season_init",type: "nominal",      title: "Region/Season/Init" },
        { field: "step",         type: "quantitative", title: "Lead time (h)" },
        { field: "value",             type: "quantitative", title: "Value", format: ".4f" },
      ],
    },
  };
}

// ---------------------------------------------------------------------------
// Legend — rendered as a zero-size chart above the table
// ---------------------------------------------------------------------------
let legendView = null;

async function renderLegend(filteredData) {
  const src = filteredData.length ? filteredData : DATA;
  const spec = {
    data: { values: src },
    config: { view: { stroke: null } },
    mark: { type: "point", filled: true, opacity: 0, size: 0 },
    width: 1,
    height: 1,
    encoding: {
      color: {
        field: "source",
        type: "nominal",
        scale: { domain: ALL_SOURCES },
        legend: {
          orient: "left", title: "Source", offset: 8,
          symbolType: "circle", symbolSize: 120,
        },
      },
      shape: {
        field: "region_season_init",
        type: "nominal",
        legend: {
          orient: "right",
          title: "Region / Season / Init",
          offset: 8,
          labelLimit: 400,
          symbolType: "circle", symbolSize: 120,
        },
      },
    },
  };
  try {
    if (legendView) { try { legendView.finalize(); } catch (e) {} }
    const result = await vegaEmbed(
      document.getElementById("legend-chart"), spec, { actions: false }
    );
    legendView = result.view;
  } catch (e) { console.warn("legend render error", e); }
}

// ---------------------------------------------------------------------------
// Main update
// ---------------------------------------------------------------------------

// Debounce so rapid filter changes don't spawn N simultaneous re-renders
let _updateTimer = null;
function scheduleUpdate() {
  clearTimeout(_updateTimer);
  _updateTimer = setTimeout(updateChart, 250);
}

// Monotonic counter: each call to updateChart increments it.
// Async cell renders check this to abort if they've been superseded.
let _epoch = 0;

async function updateChart() {
  const epoch = ++_epoch;
  console.log("[dashboard] updateChart epoch=" + epoch);

  const selRegions = getSelected("region-select");
  const selSeasons = getSelected("season-select");
  const selInits   = getSelected("init-select");
  const selSources = getSelected("source-select");
  const selMetrics = getSelected("metric-select");
  const selParams  = getSelected("param-select");

  // Filter data by region / season / init / source
  // (metric and param are handled per cell)
  let filtered = DATA;
  if (selRegions.length) filtered = filtered.filter(d => selRegions.includes(d.region));
  if (selSeasons.length) filtered = filtered.filter(d => selSeasons.includes(d.season));
  if (selInits.length)   filtered = filtered.filter(d => selInits.includes(d.init_hour));
  if (selSources.length) filtered = filtered.filter(d => selSources.includes(d.source));

  // Show / hide table columns (params)
  document.querySelectorAll("#chart-table thead th[data-param]").forEach(th => {
    th.style.display =
      (!selParams.length || selParams.includes(th.dataset.param)) ? "" : "none";
  });
  document.querySelectorAll("#chart-table tbody td[data-param]").forEach(td => {
    td.style.display =
      (!selParams.length || selParams.includes(td.dataset.param)) ? "" : "none";
  });

  // Show / hide table rows (metrics)
  document.querySelectorAll("#chart-table tbody tr[data-metric]").forEach(tr => {
    tr.style.display =
      (!selMetrics.length || selMetrics.includes(tr.dataset.metric)) ? "" : "none";
  });

  // Update legend (non-blocking)
  renderLegend(filtered);

  // Render each visible cell
  const cells = document.querySelectorAll("#chart-table .chart-cell");
  for (const el of cells) {
    const { metric, param } = el.dataset;
    const visible =
      (!selMetrics.length || selMetrics.includes(metric)) &&
      (!selParams.length  || selParams.includes(param));

    const key = cellKey(metric, param);

    if (!visible) {
      disposeView(key);
      el.innerHTML = "";
      continue;
    }

    // Dispose previous view for this cell
    disposeView(key);

    const cellData = filtered.filter(d => d.metric === metric && d.param === param);
    const spec = makeCellSpec(cellData);

    // Capture epoch for staleness check after await
    const myEpoch = epoch;
    try {
      const result = await vegaEmbed(el, spec, { actions: false });
      // If a newer updateChart fired while we were awaiting, discard this view
      if (_epoch !== myEpoch) {
        try { result.view.finalize(); } catch (e) {}
      } else {
        vegaViews.set(key, result.view);
      }
    } catch (e) {
      console.warn("cell render error", metric, param, e);
      var eb = document.getElementById("js-error-box");
      if (eb) { eb.style.display=""; eb.innerHTML += "<p>cell error [" + metric + "/" + param + "]: " + e + "</p>"; }
    }
  }

  attachZoomSync();
}

// ---------------------------------------------------------------------------
// Zoom synchronization across all chart cells
// ---------------------------------------------------------------------------
let _zoomSyncing = false;

function attachZoomSync() {
  vegaViews.forEach((view, key) => {
    view.addSignalListener("xZoom_tuple", (_name, value) => {
      if (_zoomSyncing) return;
      _zoomSyncing = true;
      vegaViews.forEach((otherView, otherKey) => {
        if (otherKey !== key) {
          try { otherView.signal("xZoom_tuple", value).run(); } catch (e) {}
        }
      });
      _zoomSyncing = false;
    });
  });
}

// ---------------------------------------------------------------------------
// Collapsible filter panel
// ---------------------------------------------------------------------------
function resizeChartScroll() {
  const scroll = document.getElementById("chart-scroll");
  const top = scroll.getBoundingClientRect().top + window.scrollY;
  scroll.style.maxHeight = `calc(100vh - ${top + 16}px)`;
}

(function () {
  const toggle  = document.getElementById("controls-toggle");
  const panel   = document.querySelector(".controls");
  const summary = document.getElementById("controls-summary");

  function updateSummary() {
    const parts = [
      getSelected("region-select"),
      getSelected("season-select"),
      getSelected("init-select"),
      getSelected("source-select"),
      getSelected("metric-select"),
      getSelected("param-select"),
    ].flatMap(v => v);
    summary.textContent = parts.join(", ");
  }

  toggle.addEventListener("click", () => {
    const collapsed = panel.classList.toggle("collapsed");
    toggle.textContent    = collapsed ? "▼ Show filters" : "▲ Hide filters";
    summary.style.display = collapsed ? "inline" : "none";
    if (collapsed) updateSummary();
    // Let the DOM reflow before measuring
    requestAnimationFrame(resizeChartScroll);
  });

  // Keep summary current when selections change (guard: some selects may be absent)
  ["region-select", "season-select", "init-select",
   "source-select", "metric-select", "param-select"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", () => {
      if (panel.classList.contains("collapsed")) updateSummary();
    });
  });

  window.addEventListener("resize", resizeChartScroll);
})();

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
updateChart();
resizeChartScroll();
