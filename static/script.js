// =============================
// 🔥 PREMIUM DASHBOARD SCRIPT
// =============================

// ---------- TOAST NOTIFICATION ----------
function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerText = message;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add("show");
    }, 100);

    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ---------- LOADING SPINNER ----------
function showLoader() {
    document.getElementById("loader").style.display = "flex";
}

function hideLoader() {
    document.getElementById("loader").style.display = "none";
}

// ---------- VIDEO PREVIEW BEFORE UPLOAD ----------
function previewVideo(input) {
    const preview = document.getElementById("videoPreview");

    if (input.files && input.files[0]) {
        const file = input.files[0];

        if (!file.type.includes("video")) {
            showToast("❌ Please upload a video file", "error");
            return;
        }

        const url = URL.createObjectURL(file);
        preview.src = url;
        preview.style.display = "block";

        showToast("🎥 Video ready for analysis", "success");
    }
}

// ---------- FORM SUBMIT WITH LOADER ----------
function handleUpload(form) {
    showLoader();
    showToast("🚀 Processing video... Please wait", "info");
    return true;
}

// ---------- WEBCAM ----------
function startWebcam() {
    showToast("📷 Starting webcam... Press Q to stop", "info");

    setTimeout(() => {
        window.location.href = "/webcam";
    }, 1000);
}

// ---------- IOT SIMULATION ----------
let iotConnected = false;

function connectIoT() {
    const status = document.querySelector(".status");

    if (!iotConnected) {
        showToast("🔌 Connecting to Arduino...", "info");

        setTimeout(() => {
            iotConnected = true;
            status.innerText = "Connected";
            status.style.color = "lightgreen";

            showToast("✅ Arduino Connected", "success");
        }, 1500);
    } else {
        showToast("Already connected!", "info");
    }
}

function triggerAlert() {
    if (!iotConnected) {
        showToast("❌ Connect IoT first!", "error");
        return;
    }

    showToast("🚨 Emergency Alert Triggered!", "error");

    // Visual effect
    document.body.style.background = "#ff4d4d";

    setTimeout(() => {
        document.body.style.background = "";
    }, 1000);
}

// ---------- DRAG & DROP UPLOAD ----------
function setupDragDrop() {
    const dropArea = document.getElementById("dropArea");

    if (!dropArea) return;

    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.classList.add("dragging");
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.classList.remove("dragging");
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();
        dropArea.classList.remove("dragging");

        const fileInput = document.querySelector("input[type='file']");
        fileInput.files = e.dataTransfer.files;

        previewVideo(fileInput);
    });
}

// ---------- INIT ----------
window.onload = () => {
    setupDragDrop();
};