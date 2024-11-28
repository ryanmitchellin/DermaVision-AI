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
		discardImage();
	}
}

// Discard image and reset the form
function discardImage() {
	const fileInput = document.getElementById("fileInput");
	const filePreview = document.getElementById("filePreview");
	const diagnoseButton = document.getElementById("diagnoseButton");
	const fileName = document.getElementById("fileName");
	const resultContainer = document.getElementById("result-container");

	fileInput.value = "";
	filePreview.classList.add("hidden");
	diagnoseButton.disabled = true;
	diagnoseButton.classList.add("disabled-btn");
	fileName.textContent = "No file chosen";

	// Hide results container
	resultContainer.classList.add("hidden");
}

// Upload image and handle the prediction process
async function uploadImage() {
	const fileInput = document.getElementById("fileInput");
	const file = fileInput.files[0];

	if (!file) {
		alert("Please upload an image.");
		return;
	}

	const formData = new FormData();
	formData.append("file", file);

	const loadingSpinner = document.getElementById("loadingSpinner");
	const resultContainer = document.getElementById("result-container");

	loadingSpinner.classList.remove("hidden");
	resultContainer.classList.add("hidden"); // Hide result container while loading

	// Simulate minimum spinner duration
	const spinnerStartTime = Date.now();

	try {
		const response = await fetch("/predict", {
			method: "POST",
			body: formData,
		});

		const result = await response.json();

		// Ensure spinner shows for at least 1 second
		const elapsedTime = Date.now() - spinnerStartTime;
		const minimumSpinnerTime = 1000;
		const remainingTime = minimumSpinnerTime - elapsedTime;

		setTimeout(() => {
			displayResults(result);
			loadingSpinner.classList.add("hidden");
		}, remainingTime > 0 ? remainingTime : 0);
	} catch (error) {
		console.error("Error:", error);
		alert("An error occurred while processing the image. Please try again.");
		loadingSpinner.classList.add("hidden"); // Hide spinner on error
	}
}

// Display results on the page
function displayResults(result) {
	const resultContainer = document.getElementById("result-container");
	const resultElement = document.getElementById("result");
	const stageElement = document.getElementById("stage");

	// Clear previous results
	resultElement.innerHTML = "";
	stageElement.innerHTML = "";

	// Display diagnosis
	resultElement.textContent = `Diagnosis: ${result.prediction}`;

	// Add "Learn more" link dynamically if the prediction is Monkeypox
	if (result.prediction.toLowerCase() === "monkeypox") {
		const diagnosisLink = document.createElement("a");
		diagnosisLink.href = "/about";
		diagnosisLink.textContent = " Learn more about Monkeypox";
		diagnosisLink.classList.add("learn-more-link");
		resultElement.appendChild(diagnosisLink);
	}

	// Display stage for Monkeypox cases
	if (result.stage) {
		stageElement.textContent = `Stage: ${result.stage}`;
	}

	// Show the results container
	resultContainer.classList.remove("hidden");
}
