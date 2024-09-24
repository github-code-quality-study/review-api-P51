import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download("vader_lexicon", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words("english"))

reviews = pd.read_csv("data/reviews.csv").to_dict("records")


class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        self.allowed_locations = [
            "Albuquerque, New Mexico",
            "Carlsbad, California",
            "Chula Vista, California",
            "Colorado Springs, Colorado",
            "Denver, Colorado",
            "El Cajon, California",
            "El Paso, Texas",
            "Escondido, California",
            "Fresno, California",
            "La Mesa, California",
            "Las Vegas, Nevada",
            "Los Angeles, California",
            "Oceanside, California",
            "Phoenix, Arizona",
            "Sacramento, California",
            "Salt Lake City, Utah",
            "San Diego, California",
            "Tucson, Arizona",
        ]

    def normalize_timestamp(self, timestamp):
        formats = [
            "%Y-%m-%d %H:%M:%S",  # Full timestamp
            "%Y-%m-%d %H:%M",  # Missing seconds
            "%Y-%m-%d",  # Missing time
        ]
        parsed_timestamp = None
        for fmt in formats:
            try:
                parsed_timestamp = datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        return parsed_timestamp

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(
        self, environ: dict[str, Any], start_response: Callable[..., Any]
    ) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Fetch the query parameters from the environ dictionary
            query_params = parse_qs(environ["QUERY_STRING"])

            # Fetch the path from the environ dictionary
            path = environ["PATH_INFO"]

            # Fetch the location from the query parameters
            location = query_params.get("location", [""])[0]

            # Fetch the timestamp from the query parameters
            start_date = query_params.get("start_date", [""])[0]
            start_date = self.normalize_timestamp(start_date)
            end_date = query_params.get("end_date", [""])[0]
            end_date = self.normalize_timestamp(end_date)

            data = reviews

            for review in data:
                review["sentiment"] = self.analyze_sentiment(review["ReviewBody"])

            if len(location) != 0 and location in self.allowed_locations:
                data = [review for review in data if review["Location"] == location]

            if start_date is not None and end_date is not None:
                data = [
                    review
                    for review in data
                    if self.normalize_timestamp(review["Timestamp"]) > start_date
                    and self.normalize_timestamp(review["Timestamp"]) < end_date
                ]

            if start_date is not None and end_date is None:
                data = [
                    review
                    for review in data
                    if self.normalize_timestamp(review["Timestamp"]) > start_date
                ]

            if start_date is None and end_date is not None:
                data = [
                    review
                    for review in data
                    if self.normalize_timestamp(review["Timestamp"]) < end_date
                ]

            # Create the response body from the reviews and convert to a JSON byte string
            response_body = json.dumps(data, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response(
                "200 OK",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body))),
                ],
            )

            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            # Read the JSON data from the request body
            request_body_size = int(environ.get("CONTENT_LENGTH", 0))
            request_body = environ["wsgi.input"].read(request_body_size)

            # Parse the request data
            try:
                request_data = parse_qs(request_body.decode("utf-8"))
            except Exception as e:
                start_response("400 Bad Request", [("Content-Length", "0")])
                return []

            # Extract the review body and location from the request data
            review_location = request_data.get("Location", [""])[0]
            if review_location not in self.allowed_locations:
                start_response("400 Bad Request", [("Content-Length", "0")])
                return []
            review_body = request_data.get("ReviewBody", [""])[0]

            if len(review_location) == 0 or len(review_body) == 0:
                start_response("400 Bad Request", [("Content-Length", "0")])
                return []

            # generate a unique id for the review
            review_id = str(uuid.uuid4())

            # generate a timestamp for the review
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create a new review dictionary
            new_review = {
                "ReviewId": review_id,
                "Location": review_location,
                "Timestamp": timestamp,
                "ReviewBody": review_body,
            }

            # Add the new review to the list of reviews
            reviews.append(new_review)

            # Add to the CSV file
            reviews_df = pd.DataFrame(reviews)
            reviews_df.to_csv("data/reviews.csv", index=False)

            # Send the response
            response_data = json.dumps(new_review, indent=2).encode("utf-8")
            start_response(
                "201 Created",
                [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_data))),
                ],
            )
            return [response_data]


if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get("PORT", 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
