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
["region-select", "season-select", "init-select",
 "source-select", "metric-select", "param-select"].forEach(id => {
  choicesInstances[id] = new Choices("#" + id, {
    searchEnabled: false,
    removeItemButton: true,
    shouldSort: false,
    itemSelectText: "",
    placeholder: false,
  });
  document.getElementById(id).addEventListener("change", scheduleUpdate);
});

function getSelected(id) {
  return choicesInstances[id].getValue(true);
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
const DATA = JSON.parse(document.getElementById("verif-data").textContent);

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
      select: { type: "interval", encodings: ["x"], zoom: "wheel![!event.shiftKey]" },
      bind: "scales",
    }],
    mark: { type: "line", point: { size: 40 } },
    width: 280,
    height: 160,
    encoding: {
      x: {
        field: "lead_time",
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
        { field: "lead_time",         type: "quantitative", title: "Lead time (h)" },
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
    mark: { type: "point", opacity: 0, size: 0 },
    width: 1,
    height: 1,
    encoding: {
      color: {
        field: "source",
        type: "nominal",
        legend: { orient: "left", title: "Source", offset: 8 },
      },
      shape: {
        field: "region_season_init",
        type: "nominal",
        legend: {
          orient: "right",
          title: "Region / Season / Init",
          offset: 8,
          labelLimit: 400,
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
// Boot
// ---------------------------------------------------------------------------
updateChart();
