import tornado.web
import logging

class BaseHandler(tornado.web.RequestHandler):
    @property
    def log(self):
        return self.application.log
    @property
    def ppid(self):
        return self.application.ppid
    @property
    def Etagger(self):
        return self.application.Etagger
    @property
    def etagger(self):
        return self.application.etagger
    @property
    def nlp(self):
        return self.application.nlp
