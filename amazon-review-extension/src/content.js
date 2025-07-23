console.log("Amazon Review Extractor Running...");

let allReviews = [];

function extractReviews() {
  document.querySelectorAll('[data-hook="review-body"]').forEach((review) => {
    allReviews.push(review.innerText.trim());
  });

  console.log(`Extracted ${allReviews.length} reviews so far...`);

  // Check if there's a "Next Page" button
  let nextButton = document.querySelector(".a-last a");
  if (nextButton && !nextButton.closest(".a-disabled")) {
    console.log("Moving to the next page...");
    nextButton.click();
    setTimeout(extractReviews, 3000); // Wait and extract again
  } else {
    console.log("Finished extracting all pages!");
    chrome.runtime.sendMessage({ reviews: allReviews });
  }
}

// Start extracting when page loads
window.onload = extractReviews;
n;
