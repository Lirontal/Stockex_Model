# apiKey =bc3e947185f24346b95ebf9e586de336

# from newsapi.sources import Sources
#
# s = Sources(API_KEY="bc3e947185f24346b95ebf9e586de336")
# print(s.get(category='technology', language='en'))
# #ars-technica

from pprint import pprint
from newsapi.articles import Articles
import webhoseio
# apiKey = "bc3e947185f24346b95ebf9e586de336"
#
# class ArticleDataCollector:
#     pass
# a = Articles(API_KEY=apiKey)
# pprint(str(a.AttrDict))
# article = a.get(source="ars-technica")
# pprint(article)

webhoseio.config("65dc4293-0ef8-4ef8-bfc9-c971a3ac8292")
a= webhoseio.query("filterWebContent", {"q" : "apple", "language" : "english"})
pprint(a)


