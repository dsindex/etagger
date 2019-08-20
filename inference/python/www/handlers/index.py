from __future__ import print_function
import sys
import os
import logging
import tornado.web
from handlers.base import BaseHandler
import json
import time

###############################################################################################
# etagger
path = os.path.dirname(os.path.abspath(__file__)) + '/lib'
sys.path.append(path)
# although `import tensorflow as tf statement is in the `input.py`,
# this statement will not be called by 
# `from handlers.index import IndexHandler, HCheckHandler, EtaggerHandler, EtaggerTestHandler`.
from input import Input
import feed

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

def build_input_feed_dict(graph, bucket, config):
    """Build input and feed_dict for bucket(inference only)
    """
    # mapping placeholders
    p_is_train = graph.get_tensor_by_name('prefix/is_train:0')
    p_sentence_length = graph.get_tensor_by_name('prefix/sentence_length:0')
    p_input_data_pos_ids = graph.get_tensor_by_name('prefix/input_data_pos_ids:0')
    p_input_data_chk_ids = graph.get_tensor_by_name('prefix/input_data_chk_ids:0')
    p_input_data_word_ids = graph.get_tensor_by_name('prefix/input_data_word_ids:0')
    p_input_data_wordchr_ids = graph.get_tensor_by_name('prefix/input_data_wordchr_ids:0')
    if 'elmo' in config.emb_class:
        p_elmo_input_data_wordchr_ids = graph.get_tensor_by_name('prefix/elmo_input_data_wordchr_ids:0')
    if 'bert' in config.emb_class:
        p_bert_input_data_token_ids = graph.get_tensor_by_name('prefix/bert_input_data_token_ids:0')
        p_bert_input_data_token_masks = graph.get_tensor_by_name('prefix/bert_input_data_token_masks:0')
        p_bert_input_data_segment_ids = graph.get_tensor_by_name('prefix/bert_input_data_segment_ids:0')

    inp = Input(bucket, config, build_output=False)
    feed_dict = {p_input_data_pos_ids: inp.example['pos_ids'],
                 p_input_data_chk_ids: inp.example['chk_ids'],
                 p_is_train: False,
                 p_sentence_length: inp.max_sentence_length}
    feed_dict[p_input_data_word_ids] = inp.example['word_ids']
    feed_dict[p_input_data_wordchr_ids] = inp.example['wordchr_ids']
    if 'elmo' in config.emb_class:
        feed_dict[p_elmo_input_data_wordchr_ids] = inp.example['elmo_wordchr_ids']
    if 'bert' in config.emb_class:
        feed_dict[p_bert_input_data_token_ids] = inp.example['bert_token_ids']
        feed_dict[p_bert_input_data_token_masks] = inp.example['bert_token_masks']
        feed_dict[p_bert_input_data_segment_ids] = inp.example['bert_segment_ids']
    return inp, feed_dict

def analyze(graph, sess, query, config, nlp):
    """Analyze query by nlp, etagger
    """
    bucket = build_bucket(nlp, query)
    inp, feed_dict = build_input_feed_dict(graph, bucket, config)
    ## mapping output/input tensors for bert
    t_bert_embeddings_subgraph = graph.get_tensor_by_name('prefix/bert_embeddings_subgraph:0')
    p_bert_embeddings = graph.get_tensor_by_name('prefix/bert_embeddings:0')
    ## mapping output tensors
    t_logits_indices = graph.get_tensor_by_name('prefix/logits_indices:0')
    t_sentence_lengths = graph.get_tensor_by_name('prefix/sentence_lengths:0')
    ## analyze
    if 'bert' in config.emb_class:
        # compute bert embedding at runtime
        bert_embeddings = sess.run([t_bert_embeddings_subgraph], feed_dict=feed_dict)
        # update feed_dict
        feed_dict[p_bert_embeddgins] = feed.align_bert_embeddings(config, bert_embeddings, inp.example['bert_wordidx2tokenidx'], -1)
    logits_indices, sentence_lengths = sess.run([t_logits_indices, t_sentence_lengths], feed_dict=feed_dict)
    tags = config.logit_indices_to_tags(logits_indices[0], sentence_lengths[0])
    ## build output
    out = []
    for i in range(len(bucket)):
        tmp = bucket[i] + ' ' + tags[i]
        tl  = tmp.split()
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

        config = self.config
        m = self.etagger[pid]
        sess = m['sess']
        graph = m['graph']
        nlp = self.nlp
        try :
            out = analyze(graph, sess, query, config, nlp)
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
        config = self.config
        m = self.etagger[pid]
        sess = m['sess']
        graph = m['graph']
        nlp = self.nlp

        if is_json_request : lines = content
        else: lines = content.split('\n')
        try:
            out_list=[]
            for line in lines :
                line = line.strip()
                if not line : continue
                out = analyze(graph, sess, line, config, nlp)
                out_list.append(out)
            self.write(dict(success=True, record=out_list, info=None))
        except Exception as e:
            msg = str(e)
            self.write(dict(success=False, info=msg))

        self.finish()
