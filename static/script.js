// script.js

document
	.getElementById("fileInput")
	.addEventListener("change", handleFileChange);

function handleFileChange() {
	const fileInput = document.getElementById("fileInput");
	const fileName = document.getElementById("fileName");
	const filePreview = document.getElementById("filePreview");
	const diagnoseButton = document.getElementById("diagnoseButton");
	const discardButton = document.querySelector(".discard-button");
	const zoomButton = document.querySelector(".zoom-button");

	if (fileInput.files.length > 0) {
		fileName.textContent = fileInput.files[0].name;
		const reader = new FileReader();
		reader.onload = function (e) {
			filePreview.src = e.target.result;
			filePreview.classList.remove("hidden");
			discardButton.classList.remove("hidden");
			zoomButton.classList.remove("hidden");
		};
		reader.readAsDataURL(fileInput.files[0]);
		diagnoseButton.disabled = false;
		diagnoseButton.classList.remove("disabled-btn");
	} else {
		fileName.textContent = "No file chosen";
		filePreview.classList.add("hidden");
		diagnoseButton.disabled = true;
		diagnoseButton.classList.add("disabled-btn");
	}
}

function showSection(sectionId) {
	document.getElementById("diagnosis").classList.add("hidden");
	document.getElementById("about").classList.add("hidden");

	document.getElementById(sectionId).classList.remove("hidden");
}

function discardImage() {
	const fileInput = document.getElementById("fileInput");
	const filePreview = document.getElementById("filePreview");
	const diagnoseButton = document.getElementById("diagnoseButton");
	const fileName = document.getElementById("fileName");
	const discardButton = document.querySelector(".discard-button");
	const zoomButton = document.querySelector(".zoom-button");

	fileInput.value = "";
	filePreview.classList.add("hidden");
	discardButton.classList.add("hidden");
	zoomButton.classList.add("hidden");
	diagnoseButton.disabled = true;
	diagnoseButton.classList.add("disabled-btn");
	fileName.textContent = "No file chosen";
}

function zoomImage() {
	const filePreview = document.getElementById("filePreview");
	const zoomedImage = document.getElementById("zoomedImage");
	const zoomModal = document.getElementById("zoomModal");

	zoomedImage.src = filePreview.src;
	zoomModal.classList.remove("hidden");

	document.addEventListener("click", outsideClickClose);
}

function outsideClickClose(event) {
	const zoomModal = document.getElementById("zoomModal");
	const zoomedImage = document.getElementById("zoomedImage");
	if (event.target === zoomModal && event.target !== zoomedImage) {
		closeModal();
	}
}

function closeModal() {
	const zoomModal = document.getElementById("zoomModal");
	zoomModal.classList.add("hidden");
	document.removeEventListener("click", outsideClickClose);
}

async function uploadImage() {
	const fileInput = document.getElementById("fileInput");
	const file = fileInput.files[0];
	if (!file) {
		alert("Please upload an image.");
		return;
	}

	const formData = new FormData();
	formData.append("file", file);

	// Show loading spinner
	document.getElementById("loadingSpinner").classList.remove("hidden");
	document.getElementById("result").classList.add("hidden");
	document.getElementById("severity").classList.add("hidden");
	document.getElementById("explanation").classList.add("hidden");

	try {
		const response = await fetch("/predict", {
			method: "POST",
			body: formData,
		});

		const result = await response.json();
		document.getElementById("result").textContent =
			"Diagnosis: " + result.prediction;
		document.getElementById("result").classList.remove("hidden");

		if (result.prediction === "Monkeypox") {
			document.getElementById("severity").textContent =
				"Severity: " + result.severity;
			document.getElementById("explanation").textContent =
				"Explanation: " + result.explanation;
			document.getElementById("severity").classList.remove("hidden");
			document.getElementById("explanation").classList.remove("hidden");
		}
	} catch (error) {
		alert("An error occurred while processing the image. Please try again.");
	} finally {
		// Hide loading spinner
		document.getElementById("loadingSpinner").classList.add("hidden");
	}
}
