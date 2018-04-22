import tensorflow as tf
import numpy as np
import grpc
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from league_pb2 import MatchPrediction, ServerStatus
from league_pb2_grpc import LeagueMatchPredictionServicer
from league_pb2_grpc import add_LeagueMatchPredictionServicer_to_server


logging.basicConfig()
logger = logging.getLogger('server')
logger.setLevel(logging.INFO)


def load_model(model_path):
  if os.path.isfile(model_path):
    logger.info('loading %s...', model_path)
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_path) as f:
      graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as g:
      tf.import_graph_def(graph_def)
      return g
  else:
    logger.info('unable to find %s', model_path)


class PredictionServicer(LeagueMatchPredictionServicer):
  def __init__(self):
    graph = load_model('./league.pb')

    self._champs = graph.get_tensor_by_name('import/champs:0')
    self._prediction = graph.get_tensor_by_name('import/output/prediction:0')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self._sess = tf.Session(config=config, graph=graph)

  def GetServerStatus(self, request, context):
    logger.info('request server status.')
    return ServerStatus(ok=True)

  def Predict(self, request, context):
    champs = np.array([[
      request.blue.top,
      request.blue.jungle,
      request.blue.mid,
      request.blue.adc,
      request.blue.support,
      request.red.top,
      request.red.jungle,
      request.red.mid,
      request.red.adc,
      request.red.support]], dtype=np.int32)
    result = self._sess.run(self._prediction, feed_dict={
      self._champs: champs
    })
    logger.info('request prediction: %s', str(result[0, :]))
    pred = MatchPrediction()
    if np.argmax(result[0, :]) == 0:
      pred.winning_team = 'blue'
    else:
      pred.winning_team = 'red'
    pred.probability = np.max(result[0, :])
    return pred

  def __del__(self):
    self._sess.close()


class PredictionServer(object):
  def __init__(self):
    logger.info('setup server...')
    self._server = grpc.server(ThreadPoolExecutor(max_workers=100))
    servicer = PredictionServicer()
    add_LeagueMatchPredictionServicer_to_server(
        servicer, self._server)
    self._server.add_insecure_port('[::]:50051')

  def start(self):
    logger.info('starting server...')
    self._server.start()
    try:
      while True:
        time.sleep(1)
    except KeyboardInterrupt:
        self._server.stop(0)


def main():
  server = PredictionServer()
  server.start()


if __name__ == '__main__':
  main()
