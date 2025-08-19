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

choicesInstances["model-select"] = new Choices("#model-select", {
    searchEnabled: false,
    removeItemButton: true,
    shouldSort: false,
    itemSelectText: "",
    placeholder: false
});
document.getElementById("model-select").addEventListener("change", updateChart);

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

// Define base spec
var spec = {
  "data": {
    "values": data
  },
  "facet": {
    "column": { "field": "param" },
    "row": { "field": "metric" }
  },
  "spec": {
    "mark": { "type": "line" },
    "encoding": {
      "x": { "field": "lead_time", "type": "ordinal" },
      "y": { "field": "value", "type": "quantitative" },
      "color": { "field": "model", "legend": { "orient": "top", "labelLimit": 1000, "symbolSize": 1000 } }
    }
  },
  "resolve": {
    "scale": {
      "y": "independent"
    }
  }
};


// Define functions

function getSelectedValues(id) {
    return choicesInstances[id].getValue(true)
}

function updateChart() {
    const selectedModels = getSelectedValues("model-select");
    const selectedparams = getSelectedValues("param-select");
    const selectedMetrics = getSelectedValues("metric-select");

    const newSpec = JSON.parse(JSON.stringify(spec));
    const filters = [];

    if (selectedModels.length > 0) {
        filters.push({ field: "model", oneOf: selectedModels });
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
