# apiKey =bc3e947185f24346b95ebf9e586de336

# from newsapi.sources import Sources
#
# s = Sources(API_KEY="bc3e947185f24346b95ebf9e586de336")
# print(s.get(category='technology', language='en'))
# #ars-technica

from pprint import pprint
from newsapi.articles import Articles
from sentiment import sentiment_text
apiKey = "bc3e947185f24346b95ebf9e586de336"
a = Articles(API_KEY=apiKey)
article = a.get(source="ars-technica")
# pprint(article)
sentiment_text(article["articles"][2]["description"])
