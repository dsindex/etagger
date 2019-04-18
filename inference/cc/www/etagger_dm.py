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
path = os.path.dirname(os.path.abspath(__file__)) + '/../wrapper'
sys.path.append(path)
import Etagger

# etagger arguments
define('so_path', default='', help='path to libetagger.so.', type=str)
define('frozen_graph_fn', default='', help='path to frozen model(ex, ./exported/ner_frozen.pb).', type=str)
define('vocab_fn', default='', help='path to vocab(ex, vocab.txt).', type=str)
define('word_length', default=15, help='max word length.', type=int)
define('lowercase', default='True', help='True if vocab file was all lowercased, otherwise False.', type=str)
define('is_memmapped', default='False', help='is memory mapped graph, True | False.', type=str)
define('num_threads', default=1, help='number of threads for tensorflow. 0 for all cores, n for n cores.', type=int)
###############################################################################################

###############################################################################################
# nlp : spacy
import spacy
###############################################################################################

from handlers.index import IndexHandler, HCheckHandler, EtaggerHandler, EtaggerTestHandler
define('port', default=8897, help='run on the given port.', type=int)
define('debug', default=True, help='run on debug mode.', type=bool)
define('process', default=3, help='number of process for service mode.', type=int)


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
        # save Etagger(python instance) for passing to handlers.
        self.Etagger = Etagger
        # create nlp(spacy) only once.
        self.nlp = spacy.load('en')
        self.log.info('initialize spacy on parent process[%s] ... done' % (ppid))
        ###############################################################################################

        log.info('http start...')

    def initialize(self) :
        pid = os.getpid()
        self.log.info('initialize per child process[%s] ...' % (pid))
        ###############################################################################################
        # create etagger instance for each child process.
        self.etagger = {}
        lowercase = False
        if options.lowercase == 'True': lowercase = True
        is_memmapped = False
        if options.is_memmapped == 'True': is_memmapped = True
        etagger = Etagger.initialize(options.so_path,
                                     options.frozen_graph_fn,
                                     options.vocab_fn,
                                     word_length=options.word_length,
                                     lowercase=lowercase,
                                     is_memmapped=is_memmapped,
                                     num_threads=options.num_threads)
        
        self.etagger[pid] = etagger
        ###############################################################################################
        self.log.info('initialize per child process[%s] ... done' % (pid))

    def finalize(self):
        # finalize resources
        self.log.info('finalize resources...')
        ## finalize something....
        for pid, etagger in self.etagger.iteritems() :
            Etagger.finalize(etagger)
        
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
