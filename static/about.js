// ABOUTT

document.addEventListener("DOMContentLoaded", () => {
	const truncatedParagraphs = document.querySelectorAll(".truncated");
	truncatedParagraphs.forEach((paragraph) => {
		if (paragraph.textContent.length > 200) {
			paragraph.dataset.fullText = paragraph.textContent;
			paragraph.textContent = paragraph.textContent.slice(0, 200) + "...";
		}
	});
});

function toggleDetails(detailsId) {
	const detailsContainer = document.getElementById(detailsId);
	const parentContainer = detailsContainer.parentElement;
	const paragraph = parentContainer.querySelector(".truncated");

	if (detailsContainer.classList.contains("hidden")) {
		detailsContainer.classList.remove("hidden");
		if (paragraph && paragraph.dataset.fullText) {
			paragraph.textContent = paragraph.dataset.fullText;
		}
	} else {
		detailsContainer.classList.add("hidden");
		if (
			paragraph &&
			paragraph.dataset.fullText &&
			paragraph.dataset.fullText.length > 200
		) {
			paragraph.textContent = paragraph.dataset.fullText.slice(0, 200) + "...";
		}
	}
}
