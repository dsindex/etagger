from __future__ import print_function
import sys
import argparse

from bert import tokenization

class Tok:
    def __init__(self, tokenizer):
        self.task = 'tok'
        self.tokenizer = tokenizer

    def __proc_bucket(self, bucket):
        for line in bucket:
            tokens = line.split()
            word = tokens[0]
            pos  = tokens[1]
            chunk = tokens[2]
            tag  = tokens[3]
            # for '-DOCSTART-'
            if word == '-DOCSTART-':
                '''
                print(word, pos, chunk, tag)
                break
                '''
                return None
            word_exts = self.tokenizer.tokenize(word)
            for m in range(len(word_exts)):
                if m == 0:
                    print(word_exts[m], pos, chunk, tag)
                else:
                    print(word_exts[m], pos, chunk, 'X')

        print('')
        return None

    def proc(self):
        bucket = []
        while 1:
            try: line = sys.stdin.readline()
            except KeyboardInterrupt: break
            if not line: break
            line = line.strip()
            if not line and len(bucket) >= 1:
                self.__proc_bucket(bucket)
                bucket = []
            if line : bucket.append(line)
        if len(bucket) != 0:
            self.__proc_bucket(bucket)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', type=str, help='path to bert vocab file', required=True)
    parser.add_argument('--do_lower_case', type=str, help='whether to lower case the input text', required=True)

    args = parser.parse_args()

    do_lower_case = True if args.do_lower_case.lower() == 'true' else False
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=do_lower_case)

    tok = Tok(tokenizer)
    tok.proc()
