import argparse
import sys

from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import six
# def analyzeText(text):
#     print("Performing sentiment analysis.")
#     API_SIZE_LIMIT = 1000000
#     text = text[:API_SIZE_LIMIT]
#     language_client = language.Client()
#     document = language_client.document_from_text(text)
#     sentiment = document.analyze_sentiment()
#
#     return sentiment

# Set the environment variable GOOGLE_APPLICATION_CREDENTIALS to C:\Users\project48\PycharmProjects\Stockex_Model\apiKeys\stockex-8badcdf0967b.json

def sentiment_text(text):
    """Detects sentiment in the text."""
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    # Instantiates a plain text document.
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects sentiment in the document. You can also analyze HTML with:
    #   document.type == enums.Document.Type.HTML
    sentiment = client.analyze_sentiment(document).document_sentiment

    print('Score: {}'.format(sentiment.score))
    print('Magnitude: {}'.format(sentiment.magnitude))

sentiment_text("ayy")