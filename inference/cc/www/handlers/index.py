from __future__ import print_function
import sys
import os
import logging
import tornado.web
from handlers.base import BaseHandler
import json
import time

###############################################################################################
# nlp : spacy
def get_entity(doc, begin, end):
    for ent in doc.ents:
        # check included
        if ent.start_char <= begin and end <= ent.end_char:
            if ent.start_char == begin: return 'B-' + ent.label_
            else: return 'I-' + ent.label_
    return 'O'
 
def build_bucket(nlp, line):
    bucket = []
    doc = nlp(line)
    for token in doc:
        begin = token.idx
        end   = begin + len(token.text) - 1
        temp = []
        temp.append(token.text)
        temp.append(token.tag_)
        temp.append('O')     # no chunking info
        entity = get_entity(doc, begin, end)
        temp.append(entity)  # entity by spacy
        temp = ' '.join(temp)
        bucket.append(temp)
    return bucket

def analyze(Etagger, etagger, nlp, query):
    """Analyze query by nlp, etagger
    """
    bucket = build_bucket(nlp, query)
    result = Etagger.analyze(etagger, bucket) 
    ## build output
    out = []
    for i in range(len(result)):
        tl = result[i]
        entry = {}
        entry['id'] = i
        entry['word'] = tl[0]
        entry['pos']  = tl[1]
        entry['chk']  = tl[2]
        entry['tag']  = tl[3]
        entry['predict']  = tl[4]
        out.append(entry)
    return out
###############################################################################################

class IndexHandler(BaseHandler):
    def get(self):
        q = self.get_argument('q', '')
        self.render('index.html', q=q)

class HCheckHandler(BaseHandler):
    def get(self):
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        templates_dir = 'templates'
        hdn_filename = '_hcheck.hdn'
        err_filename = 'error.html'
        try : fid = open(templates_dir + "/" + hdn_filename, 'r')
        except :
            self.set_status(404)
            self.render(err_filename)
        else :
            fid.close()
            self.render(hdn_filename)

class EtaggerHandler(BaseHandler):
    def get(self) :
        start_time = time.time()
        
        callback = self.get_argument('callback', '')
        mode = self.get_argument('mode', 'product')
        try :
            query = self.get_argument('q', '')
        except :
            query = "Invalid unicode in q"

        debug = {}
        debug['callback'] = callback
        debug['mode'] = mode
        pid = os.getpid()
        debug['pid'] = pid

        rst = {}
        rst['msg'] = ''
        rst['query'] = query
        if mode == 'debug' : rst['debug'] = debug

        Etagger = self.Etagger
        etagger = self.etagger[pid]
        nlp = self.nlp
        try :
            out = analyze(Etagger, etagger, nlp, query)
            rst['status'] = 200
            rst['output'] = out
        except :
            rst['status'] = 500
            rst['output'] = []
            rst['msg'] = 'analyze() fail'

        if mode == 'debug' :
            duration_time = time.time() - start_time
            debug['exectime'] = duration_time

        try :
            ret = json.dumps(rst)
        except :
            msg = "json.dumps() fail for query %s" % (query)
            self.log.debug(msg + "\n")
            err = {}
            err['status'] = 500
            err['msg'] = msg
            ret = json.dumps(err)

        if mode == 'debug' :
            self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')

        if callback.strip() :
            self.set_header('Content-Type', 'application/javascript; charset=utf-8')
            ret = 'if (typeof %s === "function") %s(%s);' % (callback, callback, ret)
        else :
            self.set_header('Content-Type', 'application/json; charset=utf-8')

        self.write(ret)
        self.finish()    
        

    def post(self):
        self.get()

class EtaggerTestHandler(BaseHandler):
    def post(self):
        if self.request.body :
            try:
                json_data = json.loads(self.request.body)
                self.request.arguments.update(json_data)
                content = ''
                if 'content' in json_data : content = json_data['content']
                is_json_request = True
            except:
                content = self.get_argument('content', "", True)
                is_json_request = False
        else:
            self.write(dict(success=False, info='no request body for post'))
            self.finish()

        pid = os.getpid()
        Etagger = self.Etagger
        etagger = self.etagger[pid]
        nlp = self.nlp

        if is_json_request : lines = content
        else: lines = content.split('\n')
        try:
            out_list=[]
            for line in lines :
                line = line.strip()
                if not line : continue
                out = analyze(Etagger, etagger, nlp, line)
                out_list.append(out)
            self.write(dict(success=True, record=out_list, info=None))
        except Exception as e:
            msg = str(e)
            self.write(dict(success=False, info=msg))

        self.finish()
