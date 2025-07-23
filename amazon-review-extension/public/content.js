console.log("Amazon Review Extractor Loaded...");

// Listen for messages from popup.js
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "startExtraction") {
    console.log("Extraction started by user.");
    extractReviews();
    sendResponse({ status: "Extraction started" });
  }
  return true;
});

let allReviews = [];

function extractReviews() {
  allReviews = [];

  // Extract product name from review page
  const productNameElem = document.querySelector('a[data-hook="product-link"]');
  const productName = productNameElem
    ? productNameElem.textContent.trim()
    : "Unknown Product";

  // Extract product image from review page
  const productImageElem = document.querySelector(
    'img[data-hook="cr-product-image"]'
  );
  const productImage = productImageElem
    ? productImageElem.getAttribute("data-a-hires") || productImageElem.src
    : "";

  function scrapePage() {
    const reviews = document.querySelectorAll('[data-hook="review-body"]');
    reviews.forEach((review) => {
      allReviews.push(review.textContent.trim());
    });

    console.log(`Extracted ${allReviews.length} reviews so far...`);

    const nextButton = document.querySelector(".a-last a");
    if (nextButton && !nextButton.closest(".a-disabled")) {
      console.log("Moving to next page...");
      nextButton.click();
      setTimeout(scrapePage, 3000);
    } else {
      console.log("Finished extraction!");
      console.log("Product:", productName);
      console.log("Image:", productImage);
      console.log("Total reviews:", allReviews.length);

      fetch("http://127.0.0.1:5000/process_reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          productName: productName,
          productImage: productImage,
          reviews: allReviews,
        }),
      })
        .then((response) => response.json())
        .then((data) => console.log("Backend response:", data))
        .catch((error) => console.error("Error:", error));
    }
  }

  scrapePage();
}
