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
        "legend": { "orient": "top", "title": "Region", "offset": 0, "padding": 10 }
      },
      "strokeDash": {
        "field": "region_season_init",
        "type": "nominal",
        "legend": null
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

  newSpec.title = "Verification using " + header;
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

  vegaEmbed('#vis', newSpec, { actions: false });
}

// Initial chart
updateChart()
