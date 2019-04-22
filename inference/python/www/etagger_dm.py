from __future__ import print_function
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
import signal
import time
import math

import tornado.web
import tornado.ioloop
import tornado.autoreload
import tornado.web
import tornado.httpserver
import tornado.process
import tornado.autoreload as autoreload
from tornado.options import define, options

###############################################################################################
# etagger
## <caution> do not `import tensorflow` before forking processes.
## see : https://github.com/tensorflow/tensorflow/issues/5448
path = os.path.dirname(os.path.abspath(__file__)) + '/../../..'
sys.path.append(path)
from embvec import EmbVec
from config import Config
# etagger arguments
define('emb_path', default='', help='path to word embedding vector + vocab(.pkl)', type=str)
define('emb_class', default='glove', help='class of embedding(glove, elmo, bert, bert+elmo)', type=str)
define('wrd_dim', default=100, help='dimension of word embedding vector', type=int)
define('word_length', default=15, help='max word length', type=int)
define('frozen_path', default='', help='path to frozen graph', type=str)
define('restore', default='', help='dummy path for config', type=str)
###############################################################################################

###############################################################################################
# nlp : spacy
import spacy
###############################################################################################

from handlers.index import IndexHandler, HCheckHandler, EtaggerHandler, EtaggerTestHandler
define('port', default=8897, help='run on the given port', type=int)
define('debug', default=True, help='run on debug mode', type=bool)
define('process', default=3, help='number of process for service mode', type=int)


log = logging.getLogger('tornado.application')

def setupAppLogger():
    fmtStr = '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    formatter = logging.Formatter(fmt=fmtStr)

    cdir = os.path.dirname(os.path.abspath(options.log_file_prefix))
    logfile = cdir + '/' + 'application.log'

    rotatingHandler = RotatingFileHandler(logfile, 'a', options.log_file_max_size, options.log_file_num_backups)
    rotatingHandler.setFormatter(formatter)
    
    if options.logging != 'none':
        log.setLevel(getattr(logging, options.logging.upper()))
    else:
        log.setLevel(logging.ERROR)

    log.propagate = False
    log.addHandler(rotatingHandler)

    return log

class Application(tornado.web.Application):
    def __init__(self):
        settings = dict(
            static_path = os.path.join(os.path.dirname(__file__), 'static'),
            template_path = os.path.join(os.path.dirname(__file__), 'templates'),
            autoescape = None,
            debug = options.debug,
            gzip = True
        )

        handlers = [
            (r'/', IndexHandler),
            (r'/_hcheck.hdn', HCheckHandler),
            (r'/etagger', EtaggerHandler),
            (r'/etaggertest', EtaggerTestHandler),
        ]

        tornado.web.Application.__init__(self, handlers, **settings)
        autoreload.add_reload_hook(self.finalize)

        self.log = setupAppLogger()
        ppid = os.getpid()
        self.ppid = ppid
        self.log.info('initialize parent process[%s] ... done' % (ppid))

        ###############################################################################################
        # create etagger config only once
        self.config = Config(options, is_training=False, emb_class=options.emb_class, use_crf=True)
        self.log.info('initialize config on parent process[%s] ... done' % (ppid))
        # create nlp(spacy) only once
        self.nlp = spacy.load('en')
        self.log.info('initialize spacy on parent process[%s] ... done' % (ppid))
        ###############################################################################################

        log.info('http start...')

    def initialize(self) :
        ###############################################################################################
        # tensorflow should be imported here for child process.
        # see : https://github.com/tensorflow/tensorflow/issues/5448
        import tensorflow as tf
        ## for LSTMBlockFusedCell(), https://github.com/tensorflow/tensorflow/issues/23369
        tf.contrib.rnn
        ## for QRNN
        try: import qrnn
        except: sys.stderr.write('import qrnn, failed\n')
        ###############################################################################################

        pid = os.getpid()
        self.log.info('initialize per child process[%s] ...' % (pid))
        ###############################################################################################
        # loading frozen model for each child process.
        self.etagger = {}
        graph = self.load_frozen_graph(tf, options.frozen_path, emb_class=self.config.emb_class)
        gpu_ops = tf.GPUOptions()
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=False,
                                      gpu_options=gpu_ops,
                                      inter_op_parallelism_threads=0,
                                      intra_op_parallelism_threads=0)
        sess = tf.Session(graph=graph, config=session_conf)
        m = {}
        m['sess'] = sess
        m['graph'] = graph
        self.etagger[pid] = m
        ###############################################################################################
        self.log.info('initialize per child process[%s] ... done' % (pid))

    def modify_op_for_elmo(self, graph_def):
        """
        reference : https://github.com/onnx/tensorflow-onnx/issues/77#issuecomment-445066091 
        """
        for node in graph_def.node:
            if node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]
        return graph_def

    def load_frozen_graph(self, tf, frozen_graph_filename, emb_class='glove', prefix='prefix'):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            if 'elmo' in emb_class:
                graph_def = self.modify_op_for_elmo(graph_def)
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def, 
                input_map=None, 
                return_elements=None, 
                op_dict=None, 
                producer_op_list=None,
                name=prefix,
            )
        return graph

    def finalize(self):
        # finalize resources
        self.log.info('finalize resources...')
        ## finalize something....
        for pid, m in self.etagger.iteritems() :
            sess = m['sess']
            sess.close()
        
        log.info('Close logger...')
        x = list(log.handlers)
        for i in x:
            log.removeHandler(i)
            i.flush()
            i.close()
        self.log.info('finalize resources... done')

def main():
    tornado.options.parse_command_line()

    '''
    # you can prefork tornado before creating application. 
    # code snippet:
    sockets = tornado.netutil.bind_sockets(options.port)
    tornado.process.fork_processes(options.process)
    application = Application()
    httpServer = tornado.httpserver.HTTPServer(application, no_keep_alive=True)
    httpServer.add_sockets(sockets) 
    '''

    application = Application()
    httpServer = tornado.httpserver.HTTPServer(application, no_keep_alive=True)
    if options.debug == True :
        httpServer.listen(options.port)
        application.initialize()
    else :
        httpServer.bind(options.port)
        if options.process == 0 :
            httpServer.start(0) # Forks multiple sub-processes, maximum to number of cores
        else :
            if options.process < 0 :
                options.process = 1
            httpServer.start(options.process) # Forks multiple sub-processes, given number
        pid = os.getpid()
        if pid != application.ppid :
            application.initialize()

    MAX_WAIT_SECONDS_BEFORE_SHUTDOWN = 3

    def sig_handler(sig, frame):
        log.warning('Caught signal: %s', sig)
        tornado.ioloop.IOLoop.instance().add_callback(shutdown)

    def shutdown():
        log.info('Stopping http server')
        httpServer.stop()

        log.info('Will shutdown in %s seconds ...', MAX_WAIT_SECONDS_BEFORE_SHUTDOWN)
        io_loop = tornado.ioloop.IOLoop.instance()

        deadline = time.time() + MAX_WAIT_SECONDS_BEFORE_SHUTDOWN

        def stop_loop():
            now = time.time()
            if now < deadline and (io_loop._callbacks or io_loop._timeouts):
                io_loop.add_timeout(now + 1, stop_loop)
            else:
                io_loop.stop()
                log.info('Shutdown')

        stop_loop()

    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    tornado.ioloop.IOLoop.instance().start()

    log.info('Exit...')

if __name__ == '__main__':
    main()
