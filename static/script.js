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

	// Display file name and preview if a file is selected
	if (fileInput.files.length > 0) {
		fileName.textContent = fileInput.files[0].name;

		// Preview the selected image
		const reader = new FileReader();
		reader.onload = function (e) {
			filePreview.src = e.target.result;
			filePreview.classList.remove("hidden");
			discardButton.classList.remove("hidden");
			zoomButton.classList.remove("hidden");
		};
		reader.readAsDataURL(fileInput.files[0]);

		// Enable the diagnose button
		diagnoseButton.disabled = false;
		diagnoseButton.classList.remove("disabled-btn");
	} else {
		discardImage();
	}
}

// Discard image and reset the form
function discardImage() {
	const fileInput = document.getElementById("fileInput");
	const filePreview = document.getElementById("filePreview");
	const diagnoseButton = document.getElementById("diagnoseButton");
	const fileName = document.getElementById("fileName");
	const discardButton = document.querySelector(".discard-button");
	const zoomButton = document.querySelector(".zoom-button");

	// Reset input and hide elements
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

	// Set modal image source and show modal
	zoomedImage.src = filePreview.src;
	zoomModal.classList.remove("hidden");

	// Add event listener for closing modal
	document.addEventListener("click", outsideClickClose);
}

// Close modal when clicking outside the zoomed image
function outsideClickClose(event) {
	const zoomModal = document.getElementById("zoomModal");
	if (event.target === zoomModal) {
		closeModal();
	}
}

// Close modal
function closeModal() {
	const zoomModal = document.getElementById("zoomModal");
	zoomModal.classList.add("hidden");
	document.removeEventListener("click", outsideClickClose);
}

// Upload image and handle the prediction process
async function uploadImage() {
	const fileInput = document.getElementById("fileInput");
	const file = fileInput.files[0];
	if (!file) {
		alert("Please upload an image.");
		return;
	}

	// Prepare form data
	const formData = new FormData();
	formData.append("file", file);

	// Show loading spinner and hide previous results
	const loadingSpinner = document.getElementById("loadingSpinner");
	loadingSpinner.classList.remove("hidden");
	document.getElementById("result").classList.add("hidden");
	document.getElementById("severity").classList.add("hidden");
	document.getElementById("explanation").classList.add("hidden");

	// Record the start time for the spinner effect
	const spinnerStartTime = Date.now();

	try {
		// Send the image to the backend
		const response = await fetch("/predict", {
			method: "POST",
			body: formData,
		});

		// Handle server response
		const result = await response.json();

		// Simulate spinner duration
		const elapsedTime = Date.now() - spinnerStartTime;
		const minimumSpinnerTime = 1000; // Minimum spinner time in ms
		const remainingTime = minimumSpinnerTime - elapsedTime;

		setTimeout(() => {
			// Display the results
			displayResults(result);

			// Hide spinner after displaying results
			loadingSpinner.classList.add("hidden");
		}, remainingTime > 0 ? remainingTime : 0);
	} catch (error) {
		// Handle errors gracefully
		console.error("Error:", error);
		alert("An error occurred while processing the image. Please try again.");
		loadingSpinner.classList.add("hidden"); // Hide spinner on error
	}
}

// Display results on the page
function displayResults(result) {
	const resultElement = document.getElementById("result");
	const severityElement = document.getElementById("severity");
	const explanationElement = document.getElementById("explanation");

	// Clear previous results
	resultElement.innerHTML = "";

	// Display diagnosis
	resultElement.textContent = "Diagnosis: " + result.prediction;
	resultElement.classList.remove("hidden");

	// Add "Learn more" link dynamically if the prediction is Monkeypox
	if (result.prediction.toLowerCase() === "monkeypox") {
		const diagnosisLink = document.createElement("a");
		diagnosisLink.href = "/about"; // Link to about.html
		diagnosisLink.textContent = ` Learn more about ${result.prediction}`;
		diagnosisLink.classList.add("learn-more-link");
		diagnosisLink.style.textDecoration = "underline"; // Ensure it looks clickable
		diagnosisLink.style.color = "orange"; // Ensure it stands out
		resultElement.appendChild(diagnosisLink); // Add the link next to the diagnosis
	}

	// Display severity and explanation for Monkeypox cases
	if (result.severity) {
		severityElement.textContent = "Severity: " + result.severity;
		severityElement.classList.remove("hidden");
	}
	if (result.explanation) {
		explanationElement.textContent = "Explanation: " + result.explanation;
		explanationElement.classList.remove("hidden");
	}
}
