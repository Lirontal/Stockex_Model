from pprint import pprint
import zmq
from StockDataCollector import StockHandler
import json
import ast
import time
from algo import Algorithm as Algo
from datetime import datetime
from StockInfoProvider import StockInfoProvider
def parse_dict(param_str):
    return ast.literal_eval(param_str)

def parse_json(param_str):
    return json.loads(param_str)

def comp(str1, str2):
    return str1.lower() == str2.lower()

class Server:
    def __init__(self, port):
        self.socket = zmq.Context().socket(zmq.REP)
        self.socket.bind("tcp://127.0.0.1:" + str(port))
        self.algo = Algo()
        self.stockHandler = self.algo.sdc
        self.sip = self.algo.sip
        self.should_shutdown = False

    def __try_communication(self):
        self.socket.send_string('{ Communication test : successful }')

    def run(self):
        try:
            while True:
                #  Wait for next request from client
                print("waiting for client input...")
                bytes = self.socket.recv()

                #  Do some 'work'
                #time.sleep(1)

                #  Send reply back to client
                message = bytes.decode('utf-8')
                print("MESSAGE: "+message)
                self.reply(json.loads(message))
                if(self.should_shutdown):
                    raise ShutdownException("server shutdown due to controller command")
                time.sleep(1)

        except Exception as e:
            if (type(e) == ShutdownException):
                print("******* Server shutting down due to controller command *******")
            else:
                print("CAUGHT AN EXCEPTION, SHUTTING DOWN...")
                s.socket.__exit__()
                raise

    def reply(self, message):
        self.socket.send_string(self.handle(message))

    def __shouldExit(self):
        self.should_shutdown = True

    def __getHistorical(self, request, entry):
        #TODO: sort requests by stock and handle requests to same stock together (not sure if faster)
        a = self.stockHandler.getHistorical(request["symbol"], request["start"], request["end"])  # add reply
        return json.dumps(json.loads(str(a)))


    def __getRecommend(self, request, entry):
        self.replies[entry] = self.algo.getRecommend()

    def _getPredHistory(self, request, entry):
        self.replies[entry] = self.algo.getPredHistory()

    def __easySearch(self, request, entry):
        a = self.algo.getEasySearch(request["budget"])

        jsonObj = {}
        i = 0
        for stock_symbol, info in list(a.items()):
            if (len(info) == 0 or len(info.iloc[0]) != 9):
                continue
            d = {}
            valid = True
            for info_entry, value in info.to_dict().items():
                if(value[self.algo.sdc.lastUpdated] != value[self.algo.sdc.lastUpdated]):
                    valid = False
                    break
                d[info_entry] = value.popitem()[1]
            if(valid):
                jsonObj[i] = d #jsonObj[stock_symbol] = d
                i += 1
        # pprint(jsonObj)
        # print(str(jsonObj)[5513:5533])
        # pprint(jsonObj)
        # pprint(jsonObj)
        return json.dumps(jsonObj)

    def __advancedSearch(self, request, entry):
        a = self.algo.getAdvSearch(request["budget"], request["companyType"], request["companyName"])
        jsonObj = {}
        i = 0
        for stock_symbol, info in list(a.items()):
            if (len(info) == 0 or len(info.iloc[0]) != 9):
                continue
            d = {}
            valid = True
            for info_entry, value in info.to_dict().items():
                if (value[self.algo.sdc.lastUpdated] != value[self.algo.sdc.lastUpdated]):
                    valid = False
                    break
                d[info_entry] = value.popitem()[1]
            if (valid):
                jsonObj[i] = d  # jsonObj[stock_symbol] = d
                i += 1
        # pprint(jsonObj)
        # print(str(jsonObj)[5513:5533])
        # pprint(jsonObj)
        # pprint(jsonObj)
        return json.dumps(jsonObj)

    def handle(self, message):
        print("Got message: ", end='')
        pprint(message)
        self.replies = {}
        #handle each request and append the replies dictionary with a reply. Exit when receiving an entry with "exit" value.
        for entry in message:
            # reply = []
            request = message[entry]
            action = request["action"]
            if (action == "exit"):
                return self.__shouldExit()
            elif(comp(action, "getRecommend")):
                return self.__getRecommend(request, entry)
            elif(comp(action, "getHistorical")):
                return self.__getHistorical(request, entry)
            elif(comp(action, "easySearch")):
                return self.__easySearch(request, entry)
            elif (comp(action, "advancedSearch")):
                return self.__advancedSearch(request, entry)
        # return self.replies

class ShutdownException(Exception): pass #define an exception for server shutdown

##############  TESTS  ##############
def test_getHistorical():
    return {"action":"getHistorical", "symbol":"GOOGL", "start":"2017-01-06", "end":"2017-01-06"}
def test_getRecommend():
    return {"action":"getRecommend"}
def test_exit():
    return {"action":"exit"}
def test(requestList):
    print("mockup request (because we have no client to ask us for a reply)...")
    i = 0
    jsonReq = {}
    for req in requestList:
        jsonReq[i] = req
        i+=1
     # , "2":"exit"
    reply = (s.handle(jsonReq))
    print("mockup reply: ")
    pprint(reply)
    ##############  /TESTS  ##############

s = Server(5555)
#test([test_getHistorical(),test_getRecommend(),test_exit()])
s.run()




