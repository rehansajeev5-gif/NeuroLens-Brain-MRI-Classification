// ==========================================
// NEUROLENS — PREMIUM IMAGE ANALYSIS UI
// Existing Flask /predict endpoint preserved
// ==========================================

const input = document.getElementById("images");
const list = document.getElementById("fileList");
const form = document.getElementById("uploadForm");
const drop = document.getElementById("dropzone");
const btn = document.getElementById("submitBtn");
const results = document.getElementById("results");
const empty = document.getElementById("empty");
const toast = document.getElementById("toast");

let selectedFiles = [];

// ---------- Notifications ----------

function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 3200);
}

// ---------- HTML escaping ----------

function esc(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[character]);
}

// ---------- Create preview container ----------

const previewGrid = document.createElement("div");
previewGrid.className = "preview-grid";
previewGrid.id = "previewGrid";

list.insertAdjacentElement("afterend", previewGrid);

// ---------- File validation ----------

function validFile(file) {
    return (
        ["image/jpeg", "image/png", "image/webp"].includes(file.type)
    );
}

// ---------- Update file input ----------

function syncInput() {
    const transfer = new DataTransfer();

    selectedFiles.forEach(file => {
        transfer.items.add(file);
    });

    input.files = transfer.files;
}

// ---------- Render image previews ----------

function renderPreviews() {
    previewGrid.innerHTML = "";

    if (!selectedFiles.length) {
        list.textContent = "No images selected";
        previewGrid.style.display = "none";
        return;
    }

    list.textContent =
        `${selectedFiles.length} image(s) ready for analysis`;

    previewGrid.style.display = "grid";

    selectedFiles.forEach((file, index) => {
        const card = document.createElement("div");
        card.className = "preview-card";

        const imageURL = URL.createObjectURL(file);

        card.innerHTML = `
            <div class="preview-image-wrap">
                <img
                    src="${imageURL}"
                    alt="Preview of ${esc(file.name)}"
                    class="preview-image"
                >

                <span class="preview-index">
                    ${(index + 1).toString().padStart(2, "0")}
                </span>

                <button
                    type="button"
                    class="remove-image"
                    aria-label="Remove ${esc(file.name)}"
                    title="Remove image"
                >×</button>
            </div>

            <div class="preview-meta">
                <span class="preview-filename">
                    ${esc(file.name)}
                </span>

                <span class="preview-size">
                    ${(file.size / (1024 * 1024)).toFixed(2)} MB
                </span>
            </div>
        `;

        const image = card.querySelector("img");

        image.addEventListener("load", () => {
            URL.revokeObjectURL(imageURL);
        });

        image.addEventListener("error", () => {
            URL.revokeObjectURL(imageURL);
        });

        card.querySelector(".remove-image").addEventListener(
            "click",
            () => {
                selectedFiles.splice(index, 1);
                syncInput();
                renderPreviews();
            }
        );

        previewGrid.appendChild(card);
    });
}

// ---------- Add selected images ----------

function addFiles(fileList) {
    const incoming = Array.from(fileList);

    const invalid = incoming.filter(file => !validFile(file));

    if (invalid.length) {
        showToast("Use JPG, PNG, or WEBP images only.");
    }

    const valid = incoming.filter(validFile);

    let added = 0;

    valid.forEach(file => {
        const duplicate = selectedFiles.some(existing =>
            existing.name === file.name &&
            existing.size === file.size &&
            existing.lastModified === file.lastModified
        );

        if (!duplicate) {
            selectedFiles.push(file);
            added++;
        }
    });

    if (added < valid.length) {
        showToast("Duplicate images were skipped.");
    }

    syncInput();
    renderPreviews();
}

input.addEventListener("change", event => {
    addFiles(event.target.files);
});

// ---------- Drag and drop ----------

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
    addFiles(event.dataTransfer.files);
});

// ---------- Scanning animation ----------

function showScanning() {
    empty.style.display = "none";

    results.innerHTML = `
        <div class="scanning-card">
            <div class="scanner-visual">
                <div class="scanner-ring ring-one"></div>
                <div class="scanner-ring ring-two"></div>
                <div class="scanner-ring ring-three"></div>

                <div class="scanner-core">N</div>
                <div class="scanner-beam"></div>
            </div>

            <div class="scanner-status">
                <span class="status-dot"></span>
                NEURAL PROCESSING ACTIVE
            </div>

            <h3>Analyzing MRI image</h3>

            <p class="scanner-description">
                Processing the selected image through the
                classification model.
            </p>

            <div class="scan-progress">
                <span></span>
            </div>

            <div class="scanner-foot">
                <span>IMAGE INFERENCE</span>
                <span class="scan-live">PROCESSING</span>
            </div>
        </div>
    `;
}

// ---------- Safe numeric formatting ----------

function numericValue(value) {
    const number = Number(value);

    return Number.isFinite(number) ? number : 0;
}

function formatScore(value) {
    return numericValue(value).toFixed(2);
}

// ---------- Build prediction cards ----------

function renderResults(data) {
    const items = Array.isArray(data.results)
        ? data.results
        : [];

    if (!items.length) {
        results.innerHTML = "";
        empty.style.display = "flex";
        showToast("No prediction results were returned.");
        return;
    }

    results.innerHTML = items.map((item, index) => {
        if (item.error) {
            return `
                <article class="result-card result-error">
                    <div class="result-card-top">
                        <span class="result-number">
                            IMAGE ${(index + 1)
                                .toString().padStart(2, "0")}
                        </span>

                        <span class="result-error-badge">
                            PROCESSING ERROR
                        </span>
                    </div>

                    <h3>${esc(item.filename || "Image")}</h3>

                    <p>${esc(item.error)}</p>
                </article>
            `;
        }

        const prediction = String(item.prediction || "Unknown");
        const confidence = numericValue(item.confidence);

        const scores = Object.entries(item.scores || {})
            .sort((a, b) => numericValue(b[1]) - numericValue(a[1]));

        const scoreRows = scores.map(([name, value]) => {
            const score = Math.max(
                0,
                Math.min(100, numericValue(value))
            );

            return `
                <div class="score-row">
                    <div class="score-labels">
                        <span>${esc(name)}</span>
                        <span>${formatScore(value)}%</span>
                    </div>

                    <div
                        class="result-score-track"
                        role="img"
                        aria-label="${esc(name)} model score:
                            ${formatScore(value)} percent"
                    >
                        <span
                            class="result-score-fill"
                            data-score="${score}"
                        ></span>
                    </div>
                </div>
            `;
        }).join("");

        return `
            <article class="result-card premium-result">
                <div class="result-card-top">
                    <span class="result-number">
                        ANALYSIS ${(index + 1)
                            .toString().padStart(2, "0")}
                    </span>

                    <span class="result-complete">
                        <span></span>
                        COMPLETE
                    </span>
                </div>

                <div class="result-main">
                    <div class="result-symbol">N.</div>

                    <div class="result-title">
                        <span class="result-overline">
                            PREDICTED CATEGORY
                        </span>

                        <h3>${esc(prediction)}</h3>

                        <p class="result-filename">
                            ${esc(item.filename || "Uploaded MRI image")}
                        </p>
                    </div>
                </div>

                <div class="result-highlight">
                    <div>
                        <span class="result-overline">
                            TOP MODEL SCORE
                        </span>

                        <p class="result-score-note">
                            Highest reported class score
                        </p>
                    </div>

                    <strong>
                        ${formatScore(confidence)}<small>%</small>
                    </strong>
                </div>
<div class="mri-visualization">
    <div class="mri-visualization-heading">
        <span class="result-overline">MODEL EXPLAINABILITY</span>
        <h4>MRI visualization</h4>
        <p>Compare the original scan with the Grad-CAM overlay.</p>
    </div>

    <div class="mri-image-grid">
        <div class="mri-image-card">
            <span class="mri-image-label">ORIGINAL MRI</span>

            ${
                item.original_image
                    ? `<img
                        src="${esc(item.original_image)}"
                        alt="Original MRI scan"
                        class="mri-result-image"
                        loading="lazy"
                    >`
                    : `<p class="mri-image-missing">
                        Original image unavailable
                    </p>`
            }
        </div>

        <div class="mri-image-card">
            <span class="mri-image-label">GRAD-CAM OVERLAY</span>

            ${
                item.heatmap_image
                    ? `<img
                        src="${esc(item.heatmap_image)}"
                        alt="Grad-CAM overlay highlighting image regions influencing the model"
                        class="mri-result-image"
                        loading="lazy"
                    >`
                    : `<p class="mri-image-missing">
                        Heatmap unavailable
                    </p>`
            }
        </div>
    </div>

    <p class="mri-heatmap-note">
        The overlay highlights regions influencing the model's output.
        It is not a tumor boundary or a medical diagnosis.
    </p>
</div>            
                <div class="result-divider"></div>

                <div class="score-heading">
                    <div>
                        <span class="result-overline">
                            CLASS DISTRIBUTION
                        </span>

                        <h4>Model output scores</h4>
                    </div>

                    <span class="score-count">
                        ${scores.length} CLASSES
                    </span>
                </div>

                <div class="score-list">
                    ${scoreRows}
                </div>

                <div class="result-disclaimer">
                    <span>ⓘ</span>

                    <p>
                        Educational output only. Model scores do not
                        establish a diagnosis or rule out disease.
                        Consult a qualified medical professional.
                    </p>
                </div>
            </article>
        `;
    }).join("");

    empty.style.display = "none";

    // Animate score bars after the cards have been rendered.
    requestAnimationFrame(() => {
        results.querySelectorAll(".result-score-fill")
            .forEach((bar, index) => {
                const score = Number(bar.dataset.score) || 0;

                setTimeout(() => {
                    bar.style.width = `${score}%`;
                }, 100 + index * 70);
            });
    });
}

// ---------- Analyze uploaded images ----------

form.addEventListener("submit", async event => {
    event.preventDefault();

    if (!selectedFiles.length) {
        showToast("Choose an MRI image first.");
        return;
    }

    const data = new FormData();

    selectedFiles.forEach(file => {
        data.append("images", file);
    });

    btn.disabled = true;
    btn.innerHTML = `
        <span>Analyzing images...</span>
        <span class="button-spinner">↻</span>
    `;

    showScanning();

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: data
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                result.error || "Prediction request failed."
            );
        }

        renderResults(result);

        if (result.results?.some(item => item.error)) {
            showToast("Some images could not be processed.");
        }
    } catch (error) {
        results.innerHTML = "";
        empty.style.display = "flex";

        showToast(
            error.message || "Unable to connect to the server."
        );
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
            <span>Analyze image</span>
            <span>→</span>
        `;
    }
});

// ---------- Initial state ----------

renderPreviews();