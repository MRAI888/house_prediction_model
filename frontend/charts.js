document.addEventListener("DOMContentLoaded", () => {
    renderMetricsChart();
    renderPredictionSummaryChart();
    renderDriftChart();
});

function renderMetricsChart() {
    const canvas = document.getElementById("metricsChart");
    if (!canvas) return;

    new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels: ["Linear Reg", "Decision Tree", "Random Forest", "Gradient Boost", "XGBoost"],
            datasets: [
                {
                    label: "R² Score",
                    data: [0.9595, 0.9972, 0.9990, 0.9883, 0.9959],
                    backgroundColor: ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8"],
                    borderRadius: 8
                },
                {
                    label: "MAPE (%)",
                    data: [13.14, 1.78, 1.44, 6.45, 3.93],
                    backgroundColor: ["#fde68a", "#fcd34d", "#fbbf24", "#f59e0b", "#d97706"],
                    borderRadius: 8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function renderPredictionSummaryChart() {
    const canvas = document.getElementById("confusionChart");
    if (!canvas) return;

    new Chart(canvas.getContext("2d"), {
        type: "doughnut",
        data: {
            labels: ["Low (<£150k)", "Mid (£150k–£400k)", "High (>£400k)"],
            datasets: [{
                data: [18, 46, 36],
                backgroundColor: ["#10b981", "#f59e0b", "#2563eb"],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom"
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${ctx.raw}%`
                    }
                }
            }
        }
    });
}

function renderDriftChart() {
    const canvas = document.getElementById("driftChart");
    if (!canvas) return;

    const psiValues = [0.08, 0.12, 0.05, 0.09, 0.15];
    const labels = ["Price", "New Build", "Property Type", "Sales Volume", "Location Encoding"];

    new Chart(canvas.getContext("2d"), {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "PSI",
                data: psiValues,
                backgroundColor: psiValues.map(v => {
                    if (v < 0.10) return "#10b981";
                    if (v < 0.20) return "#f59e0b";
                    return "#ef4444";
                }),
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `PSI: ${ctx.raw.toFixed(2)}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 0.25
                }
            }
        },
        plugins: [thresholdLinePlugin]
    });
}

const thresholdLinePlugin = {
    id: "thresholdLinePlugin",
    afterDraw(chart) {
        const { ctx, chartArea, scales } = chart;
        if (!chartArea || !scales.y) return;

        const threshold = 0.10;
        const y = scales.y.getPixelForValue(threshold);

        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([6, 6]);
        ctx.moveTo(chartArea.left, y);
        ctx.lineTo(chartArea.right, y);
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#dc2626";
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#dc2626";
        ctx.font = "12px Arial";
        ctx.fillText("Warning threshold (0.10)", chartArea.left + 8, y - 8);
        ctx.restore();
    }
};