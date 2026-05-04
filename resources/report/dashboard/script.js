// Tab switching
document.querySelectorAll(".tab-link").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab-link").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
    button.classList.add("active");
    document.getElementById(button.dataset.tab).classList.add("active");
  });
});


// Initialize selection widgets
const choicesInstances = {};

choicesInstances["region-select"] = new Choices("#region-select", {
  searchEnabled: false,
  removeItemButton: true,
  shouldSort: false,
  itemSelectText: "",
  placeholder: false
});
document.getElementById("region-select").addEventListener("change", updateChart);

choicesInstances["season-select"] = new Choices("#season-select", {
  searchEnabled: false,
  removeItemButton: true,
  shouldSort: false,
  itemSelectText: "",
  placeholder: false
});
document.getElementById("season-select").addEventListener("change", updateChart);

choicesInstances["init-select"] = new Choices("#init-select", {
  searchEnabled: false,
  removeItemButton: true,
  shouldSort: false,
  itemSelectText: "",
  placeholder: false
});
document.getElementById("init-select").addEventListener("change", updateChart);

choicesInstances["source-select"] = new Choices("#source-select", {
  searchEnabled: false,
  removeItemButton: true,
  shouldSort: false,
  itemSelectText: "",
  placeholder: false
});
document.getElementById("source-select").addEventListener("change", updateChart);

choicesInstances["metric-select"] = new Choices("#metric-select", {
  searchEnabled: false,
  removeItemButton: true,
  shouldSort: false,
  itemSelectText: "",
  placeholder: false
});
document.getElementById("metric-select").addEventListener("change", updateChart);

choicesInstances["param-select"] = new Choices("#param-select", {
  searchEnabled: false,
  removeItemButton: true,
  shouldSort: false,
  itemSelectText: "",
  placeholder: false
});
document.getElementById("param-select").addEventListener("change", updateChart);

// Get the data (embedded in the HTML)
data = JSON.parse(document.getElementById("verif-data").textContent)
header = document.getElementById("header-text").textContent.trim()

// Define base spec
var spec = {
  "data": { "values": data },
  "config": {
    "scale": { "continuousPadding": 1 }
  },
  "facet": {
    "row": { "field": "metric", "type": "nominal", "title": null },
    "column": { "field": "param", "type": "nominal" , "title": null },
  },
  "resolve": {
    "scale": {
      "x": "shared",
      "y": "independent"
    },
  },
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
    "mark": {"type": "line", "point": { "size": 50 } },
    "width": 300,
    "height": 200,
    "encoding": {
      "x": {
        "field": "lead_time",
        "type": "quantitative"
      },
      "y": {
        "field": "value",
        "type": "quantitative" ,
          "scale": { "zero": false }
      },
      "color": {
        "field": "source",
        "type": "nominal",
        "legend": { "orient": "top", "title": "Data Source", "offset": 0, "padding": 10 }
      },
      "shape": {
        "field": "region_season_init",
        "type": "nominal",
        "legend": { "orient": "top", "title": "Region, Season, Initialization", "offset": 0, "padding": 10 }
      },
      "strokeDash": {
        "field": "region_season_init",
        "type": "nominal",
        "legend": { "orient": "top", "title": "Region, Season, Initialization", "offset": 0, "padding": 10 }
      },
      "tooltip": [
        { "field": "region", "type": "nominal", "title": "Region" },
        { "field": "source", "type": "nominal", "title": "Source" },
        { "field": "param", "type": "nominal", "title": "Parameter" },
        { "field": "metric", "type": "nominal", "title": "Metric" },
        { "field": "lead_time", "type": "quantitative", "title": "Lead Time (h)" },
        { "field": "value", "type": "quantitative", "title": "Value" }
      ]
    },
  },
};


// Define functions

function getSelectedValues(id) {
  return choicesInstances[id].getValue(true)
}

function updateChart() {
  const selectedRegions = getSelectedValues("region-select");
  const selectedSeasons = getSelectedValues("season-select");
  const selectedInits = getSelectedValues("init-select");
  const selectedSources = getSelectedValues("source-select");
  const selectedparams = getSelectedValues("param-select");
  const selectedMetrics = getSelectedValues("metric-select");

  const newSpec = JSON.parse(JSON.stringify(spec));
  const filters = [];

  newSpec.title = header;
  if (selectedRegions.length > 0) {
    filters.push({ field: "region", oneOf: selectedRegions });
  }
  if (selectedSeasons.length > 0) {
    filters.push({ field: "season", oneOf: selectedSeasons });
  }
  if (selectedInits.length > 0) {
    filters.push({ field: "init_hour", oneOf: selectedInits });
  }
  if (selectedSources.length > 0) {
    filters.push({ field: "source", oneOf: selectedSources });
  }
  if (selectedparams.length > 0) {
    filters.push({ field: "param", oneOf: selectedparams });
  }
  if (selectedMetrics.length > 0) {
    filters.push({ field: "metric", oneOf: selectedMetrics });
  }

  if (filters.length > 0) {
    newSpec.transform = [{ filter: { and: filters } }];
  }

  vegaEmbed('#vis', newSpec, { actions: false }).then(() => {
    // Small delay to ensure SVG is fully laid out before reading positions
    setTimeout(setupStickyHeaders, 100);
  });
}

function setupStickyHeaders() {
  const visOuter = document.getElementById('vis-outer');
  const vis = document.getElementById('vis');
  const svg = vis.querySelector('svg');
  if (!svg) return;

  const colBar = document.getElementById('col-sticky-bar');
  const rowBar = document.getElementById('row-sticky-bar');
  const corner = document.getElementById('sticky-corner');

  colBar.innerHTML = '';
  rowBar.innerHTML = '';

  const outerRect = visOuter.getBoundingClientRect();

  // --- Column headers (param names, top of chart) ---
  const colTexts = svg.querySelectorAll('g.mark-group.role-column-header g.mark-text text');
  let colBarHeight = 0;

  colTexts.forEach(el => {
    const r = el.getBoundingClientRect();
    if (!r.width) return;
    const span = document.createElement('div');
    span.textContent = el.textContent.trim();
    span.style.cssText = `
      position: absolute;
      left: ${r.left - outerRect.left + visOuter.scrollLeft}px;
      width: ${r.width}px;
      text-align: center;
      white-space: nowrap;
      padding: 2px 4px;
      background: white;
    `;
    colBar.appendChild(span);
    if (r.height + 4 > colBarHeight) colBarHeight = r.height + 4;
  });
  colBar.style.height = colBarHeight + 'px';

  // --- Row headers (metric names, left/right of chart) ---
  const rowTexts = svg.querySelectorAll('g.mark-group.role-row-header g.mark-text text');
  let rowBarWidth = 0;

  rowTexts.forEach(el => {
    const r = el.getBoundingClientRect();
    if (!r.height) return;
    const div = document.createElement('div');
    div.textContent = el.textContent.trim();
    div.style.cssText = `
      position: absolute;
      top: ${r.top - outerRect.top + visOuter.scrollTop}px;
      height: ${r.height}px;
      line-height: ${r.height}px;
      white-space: nowrap;
      padding: 0 4px;
      background: white;
    `;
    rowBar.appendChild(div);
    if (r.width + 8 > rowBarWidth) rowBarWidth = r.width + 8;
  });
  rowBar.style.width = rowBarWidth + 'px';

  corner.style.width = rowBarWidth + 'px';
  corner.style.height = colBarHeight + 'px';

  // --- Scroll handler: move bars to track scroll ---
  visOuter.removeEventListener('scroll', visOuter._stickyHandler);
  visOuter._stickyHandler = () => {
    colBar.style.top = visOuter.scrollTop + 'px';
    rowBar.style.left = visOuter.scrollLeft + 'px';
    corner.style.top = visOuter.scrollTop + 'px';
    corner.style.left = visOuter.scrollLeft + 'px';
  };
  visOuter.addEventListener('scroll', visOuter._stickyHandler);
}

// Initial chart
updateChart()
