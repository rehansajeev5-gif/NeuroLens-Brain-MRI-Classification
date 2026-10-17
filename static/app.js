const input = document.getElementById("images");
const list = document.getElementById("fileList");
const fileCount = document.getElementById("fileCount");
const filePreview = document.getElementById("filePreview");

const form = document.getElementById("uploadForm");
const drop = document.getElementById("dropzone");
const btn = document.getElementById("submitBtn");

const results = document.getElementById("results");
const empty = document.getElementById("empty");
const toast = document.getElementById("toast");
const analysisStatus = document.getElementById("analysisStatus");

let toastTimer;
let previewUrls = [];
let latestResults = [];


/* Notifications */

function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");

    clearTimeout(toastTimer);

    toastTimer = setTimeout(() => {
        toast.classList.remove("show");
    }, 3200);
}


/* HTML escaping */

function esc(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[character]);
}


/* Clean up image preview URLs */

function clearPreviews() {
    previewUrls.forEach(url => URL.revokeObjectURL(url));
    previewUrls = [];
}


/* Update the selected files */

function updateFiles() {
    clearPreviews();

    const files = Array.from(input.files);

    fileCount.textContent = files.length;
    filePreview.replaceChildren();

    if (!files.length) {
        list.textContent = "No images selected";
        return;
    }

    list.textContent = files
        .map(file => file.name)
        .join(", ");

    files.forEach(file => {
        if (!file.type.startsWith("image/")) {
            return;
        }

        const url = URL.createObjectURL(file);
        previewUrls.push(url);

        const item = document.createElement("div");
        item.className = "preview-item";

        const image = document.createElement("img");
        image.src = url;
        image.alt = `Preview of ${file.name}`;

        const filename = document.createElement("span");
        filename.textContent = file.name;
        filename.title = file.name;

        item.append(image, filename);
        filePreview.appendChild(item);
    });
}

input.addEventListener("change", updateFiles);


/* Drag and drop */

["dragenter", "dragover"].forEach(eventName => {
    drop.addEventListener(eventName, event => {
        event.preventDefault();
        drop.classList.add("drag");
    });
});

["dragleave", "drop"].forEach(eventName => {
    drop.addEventListener(eventName, event => {
        event.preventDefault();
        drop.classList.remove("drag");
    });
});

drop.addEventListener("drop", event => {
    const files = Array.from(event.dataTransfer.files);

    if (!files.length) {
        return;
    }

    const allowedExtensions = /\.(jpe?g|png|webp)$/i;

    const validFiles = files.filter(file => {
        return allowedExtensions.test(file.name);
    });

    if (!validFiles.length) {
        showToast("Choose JPG, PNG, or WEBP images.");
        return;
    }

    const transfer = new DataTransfer();

    validFiles.forEach(file => transfer.items.add(file));

    input.files = transfer.files;

    updateFiles();

    if (validFiles.length !== files.length) {
        showToast("Some unsupported files were skipped.");
    }
});


/* Button states */

function setAnalyzing(isAnalyzing) {
    btn.disabled = isAnalyzing;

    if (isAnalyzing) {
        btn.innerHTML = `
            <span class="button-content">
                <span class="button-icon">◌</span>
                Analyzing images...
            </span>
            <span class="button-arrow">↻</span>
        `;

        analysisStatus.textContent =
            "Processing images with the classification model...";

        analysisStatus.classList.add("active");
    } else {
        btn.innerHTML = `
            <span class="button-content">
                <span class="button-icon">✳</span>
                Analyze images
            </span>
            <span class="button-arrow">↗</span>
        `;

        analysisStatus.classList.remove("active");
        analysisStatus.textContent = "";
    }
}


/* Render prediction results */

function renderResults(items) {
    results.replaceChildren();

    if (!items || !items.length) {
        empty.style.display = "flex";
        showToast("The model returned no results.");
        return;
    }

    empty.style.display = "none";

    items.forEach((item, index) => {
        const card = document.createElement("article");
        card.className = "result-card";
        card.style.animationDelay = `${index * 100}ms`;

        if (item.error) {
            card.innerHTML = `
                <div class="result-head">
                    <h3>${esc(item.filename || "Image")}</h3>
                </div>
                <p class="disclaimer">${esc(item.error)}</p>
            `;

            results.appendChild(card);
            return;
        }

        const scores = Object.entries(item.scores || {})
            .sort((a, b) => Number(b[1]) - Number(a[1]));

        const safeScores = scores.map(([name, value]) => {
            const numericValue = Number(value);

            return [
                name,
                Number.isFinite(numericValue)
                    ? Math.max(0, Math.min(100, numericValue))
                    : 0
            ];
        });

        card.innerHTML = `
            <div class="result-head">
                <h3>${esc(item.filename || "Uploaded image")}</h3>
                <span class="prediction">
                    ${esc(item.prediction || "Unknown")}
                </span>
            </div>

            <div class="confidence">
                Top model score:
                <b>${esc(item.confidence)}%</b>
            </div>

            ${safeScores.map(([name, value]) => `
                <div class="score">
                    <div class="scoreline">
                        <span>${esc(name)}</span>
                        <span>${value.toFixed(2)}%</span>
                    </div>

                    <div class="bar">
                        <span data-score="${value}"></span>
                    </div>
                </div>
            `).join("")}

            <div class="disclaimer">
                Educational output only. This prediction is not a diagnosis
                and cannot reliably rule out disease.
            </div>
        `;

        results.appendChild(card);
    });

    /* Animate score bars after the cards are rendered */

    requestAnimationFrame(() => {
        results.querySelectorAll(".bar span").forEach((bar, index) => {
            setTimeout(() => {
                bar.style.width = `${bar.dataset.score}%`;
            }, Math.min(index * 45, 350));
        });
    });

    results.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
}


/* Submit images to the existing Flask backend */

form.addEventListener("submit", async event => {
    event.preventDefault();

    if (!input.files.length) {
        showToast("Choose an image first.");
        return;
    }

    const files = Array.from(input.files);
    const allowedExtensions = /\.(jpe?g|png|webp)$/i;

    if (files.some(file => !allowedExtensions.test(file.name))) {
        showToast("Only JPG, PNG, and WEBP images are supported.");
        return;
    }

    const data = new FormData();

    files.forEach(file => {
        data.append("images", file);
    });

    latestResults = [];

    results.replaceChildren();
    empty.style.display = "none";

    setAnalyzing(true);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: data
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                result.error || "The prediction request failed."
            );
        }

        if (!Array.isArray(result.results)) {
            throw new Error("The server returned an unexpected response.");
        }

        latestResults = result.results;

        renderResults(latestResults);

        showToast("Analysis finished.");

    } catch (error) {
        empty.style.display = "flex";

        showToast(
            error.message || "Something went wrong during analysis."
        );

    } finally {
        setAnalyzing(false);
    }
});


/* Download the latest results as a CSV report */

function downloadReport() {
    if (!latestResults.length) {
        showToast("Analyze an image before downloading a report.");
        return;
    }

    const rows = [
        [
            "Filename",
            "Predicted category",
            "Top model score (%)",
            "Class",
            "Model score (%)",
            "Notice"
        ]
    ];

    latestResults.forEach(item => {
        if (item.error) {
            rows.push([
                item.filename || "",
                "Error",
                "",
                "",
                "",
                item.error
            ]);

            return;
        }

        const scores = Object.entries(item.scores || {});

        if (!scores.length) {
            rows.push([
                item.filename || "",
                item.prediction || "",
                item.confidence ?? "",
                "",
                "",
                "Educational output only; not a medical diagnosis."
            ]);

            return;
        }

        scores.forEach(([className, score]) => {
            rows.push([
                item.filename || "",
                item.prediction || "",
                item.confidence ?? "",
                className,
                score,
                "Educational output only; not a medical diagnosis."
            ]);
        });
    });

    const csv = rows.map(row => {
        return row.map(value => {
            const text = String(value ?? "");

            return `"${text.replace(/"/g, '""')}"`;
        }).join(",");
    }).join("\r\n");

    const blob = new Blob(
        ["\uFEFF" + csv],
        { type: "text/csv;charset=utf-8;" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "neurolens-results.csv";

    document.body.appendChild(link);
    link.click();
    link.remove();

    setTimeout(() => URL.revokeObjectURL(url), 1000);

    showToast("Results report downloaded.");
}


/* Add the report button after results are rendered */

const originalRenderResults = renderResults;

renderResults = function(items) {
    originalRenderResults(items);

    if (!items || !items.length) {
        return;
    }

    if (!document.getElementById("downloadReportBtn")) {
        const reportButton = document.createElement("button");

        reportButton.id = "downloadReportBtn";
        reportButton.type = "button";
        reportButton.className = "primary report-button";

        reportButton.style.marginTop = "14px";

        reportButton.innerHTML = `
            <span class="button-content">
                <span class="button-icon">↓</span>
                Download results report
            </span>
            <span class="button-arrow">↗</span>
        `;

        reportButton.addEventListener("click", downloadReport);

        results.appendChild(reportButton);
    }
};


/* Clean up preview URLs when leaving the page */

window.addEventListener("beforeunload", clearPreviews);