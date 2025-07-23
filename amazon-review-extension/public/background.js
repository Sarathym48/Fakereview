chrome.runtime.onInstalled.addListener(() => {
  console.log("Amazon Review Extractor Extension Installed");
});

// Listen for messages from content or popup scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "fetch_reviews") {
    console.log("Fetching reviews...");
    sendResponse({ status: "Processing reviews..." });
  }
});
