import redis
import json

PYTHON2QT = 'PYTHON2QT'
QT2PYTHON = 'QT2PYTHON'


class PubSub(object):
    def __init__(self, host, port, db):
        self.__conn = redis.Redis(host, port, db)

    def publish(self, channel, msg):
        self.__conn.publish(channel, msg)
        return True

    def subscribe(self, channel):
        pub = self.__conn.pubsub()
        pub.subscribe(channel)
        pub.parse_response()
        return pub

if __name__ == '__main__':

    obj = PubSub('localhost', 6379, 1)
    redis_sub = obj.subscribe(QT2PYTHON)
    # while True:
    #     msg = redis_sub.parse_response()
    #     msg = msg[2].decode()
    #     my_json = json.loads(msg)
    #     print(my_json['entityInfo'])
    #
    #     decisions = {}
    #     k = 100
    #     for Id in my_json.keys():
    #         decisions[Id] = {'height': k, 'fight': k}
    #         k += 1
    decisions={'actionInfo': [{'id': 1001, 'xforce': 0.6637248, 'yforce': -0.01765621}, {'id': 1002, 'xforce': -0.07543644000000001, 'yforce': 0.10804170999999999}, {'id': 1003, 'xforce': -0.25735593999999995, 'yforce': 0.57929251}, {'id': 1004, 'xforce': 0.036482449999999986, 'yforce': 0.42634271}, {'id': 1005, 'xforce': 0.66637614, 'yforce': -0.003634749999999999}, {'id': 1006, 'xforce': -0.59426338, 'yforce': -0.12550427}]}

    obj.publish(PYTHON2QT, json.dumps(decisions))