// script.js

// Add event listener to file input
document
	.getElementById("fileInput")
	.addEventListener("change", handleFileChange);

// Handle file input change
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

// Toggle sections (for future expansions like About page)
function showSection(sectionId) {
	document.getElementById("diagnosis").classList.add("hidden");
	document.getElementById("about").classList.add("hidden");

	document.getElementById(sectionId).classList.remove("hidden");
}

// Discard image and reset the form
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

// Zoom image modal functionality
function zoomImage() {
	const filePreview = document.getElementById("filePreview");
	const zoomedImage = document.getElementById("zoomedImage");
	const zoomModal = document.getElementById("zoomModal");

	zoomedImage.src = filePreview.src;
	zoomModal.classList.remove("hidden");

	document.addEventListener("click", outsideClickClose);
}

// Close modal when clicking outside
function outsideClickClose(event) {
	const zoomModal = document.getElementById("zoomModal");
	const zoomedImage = document.getElementById("zoomedImage");
	if (event.target === zoomModal && event.target !== zoomedImage) {
		closeModal();
	}
}

// Close modal
function closeModal() {
	const zoomModal = document.getElementById("zoomModal");
	zoomModal.classList.add("hidden");
	document.removeEventListener("click", outsideClickClose);
}

// Upload image and show spinner effect during processing
async function uploadImage() {
	const fileInput = document.getElementById("fileInput");
	const file = fileInput.files[0];
	if (!file) {
		alert("Please upload an image.");
		return;
	}

	const formData = new FormData();
	formData.append("file", file);

	// Show loading spinner and hide other elements
	const loadingSpinner = document.getElementById("loadingSpinner");
	loadingSpinner.classList.remove("hidden");
	document.getElementById("result").classList.add("hidden");
	document.getElementById("severity").classList.add("hidden");
	document.getElementById("explanation").classList.add("hidden");

	// Record the start time for the spinner
	const spinnerStartTime = Date.now();

	try {
		// Send the image to the server
		const response = await fetch("/predict", {
			method: "POST",
			body: formData,
		});

		const result = await response.json();

		// Wait for at least 1 second before showing the results
		const elapsedTime = Date.now() - spinnerStartTime;
		const minimumSpinnerTime = 1000; // 1 second
		const remainingTime = minimumSpinnerTime - elapsedTime;

		setTimeout(
			() => {
				// Display the diagnosis result
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

				// Add a link to the about page for the specific diagnosis
				const diagnosisLink = document.createElement("a");
				diagnosisLink.href = `/about#${result.prediction
					.toLowerCase()
					.replace(" ", "-")}-card-details`;
				diagnosisLink.textContent = `Learn more about ${result.prediction}`;
				diagnosisLink.classList.add("learn-more-link");
				document.getElementById("result").appendChild(diagnosisLink);

				// Hide the spinner after displaying results
				loadingSpinner.classList.add("hidden");
			},
			remainingTime > 0 ? remainingTime : 0
		);
	} catch (error) {
		// Handle errors gracefully
		alert("An error occurred while processing the image. Please try again.");
		loadingSpinner.classList.add("hidden"); // Hide spinner if error occurs
	}
}
