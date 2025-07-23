import requests
import json

# Define the Flask API endpoint
url = "http://127.0.0.1:5000/process_reviews"

# Sample reviews for testing
sample_reviews = {
    "reviews": [
        "This product is amazing! I love it so much.",
        "Absolutely terrible. It broke within two days.",
        "Decent quality, but a bit overpriced.",
    ]
}

# Send POST request
response = requests.post(url, json=sample_reviews)

# Print the response
if response.status_code == 200:
    print("✅ Success! Response from API:")
    print(json.dumps(response.json(), indent=4))  # Pretty print JSON response
else:
    print(f"❌ Error {response.status_code}: {response.text}")
