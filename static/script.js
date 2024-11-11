// script.js

document
	.getElementById("fileInput")
	.addEventListener("change", handleFileChange);

function handleFileChange() {
	const fileInput = document.getElementById("fileInput");
	const fileName = document.getElementById("fileName");
	const filePreview = document.getElementById("filePreview");
	const diagnoseButton = document.getElementById("diagnoseButton");

	if (fileInput.files.length > 0) {
		fileName.textContent = fileInput.files[0].name;
		const reader = new FileReader();
		reader.onload = function (e) {
			filePreview.src = e.target.result;
			filePreview.classList.remove("hidden");
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
