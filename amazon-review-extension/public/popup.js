document.addEventListener("DOMContentLoaded", () => {
  console.log("[Popup.js] Popup loaded!"); // Debugging log
  document.getElementById("startButton").addEventListener("click", () => {
    console.log("[Popup.js] Button clicked, sending message..."); // Debugging log
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs.length > 0) {
        chrome.tabs.sendMessage(
          tabs[0].id,
          { action: "startExtraction" },
          (response) => {
            console.log("[Popup.js] Message sent, response:", response);
          }
        );
      } else {
        console.error("[Popup.js] No active tab found!");
      }
    });
  });
});
