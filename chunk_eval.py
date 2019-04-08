from __future__ import print_function
import sys
import argparse

class ChunkEval:
    """Chunk-based evaluation
    """

    def __init__(self):
        self.tag_sents = []
        self.pred_sents = []

    def __eval_bucket(self, bucket):
        tag_sent = []
        pred_sent = []
        line_num = 0
        for line in bucket:
            tokens = line.split()
            size = len(tokens)
            if line_num == 0 and size == 3: # skip 'USING SKIP CONNECTIONS'
                line_num += 1
                continue
            assert(size == 5)
            w = tokens[0]
            pos = tokens[1]
            chunk = tokens[2]
            tag = tokens[3]
            pred = tokens[4]
            tag_sent.append(tag)
            pred_sent.append(pred)
            line_num += 1
        self.tag_sents.append(tag_sent)
        self.pred_sents.append(pred_sent)

    def eval(self):
        """Compute micro chunk fscore along with precision, recall given file.
        """
        bucket = []
        while 1:
            try: line = sys.stdin.readline()
            except KeyboardInterrupt: break
            if not line: break
            line = line.strip()
            if not line and len(bucket) >= 1:
                self.__eval_bucket(bucket)
                bucket = []
            if line : bucket.append(line)
        if len(bucket) != 0:
            self.__eval_bucket(bucket)
        fscore = self.compute_f1(self.pred_sents, self.tag_sents)
        print('precision, recall, fscore = ', fscore)

    @staticmethod
    def compute_precision(guessed_sentences, correct_sentences):
        """Compute micro precision given tag-predictions(guessed sentences)
        and tag-corrects(correct sentences).
        """
        assert(len(guessed_sentences) == len(correct_sentences))
        correctCount = 0
        count = 0
        for sentenceIdx in range(len(guessed_sentences)):
            guessed = guessed_sentences[sentenceIdx]
            correct = correct_sentences[sentenceIdx]
            assert(len(guessed) == len(correct))
            idx = 0
            while idx < len(guessed):
                if guessed[idx][0] == 'B': # A new chunk starts
                    count += 1
                    '''
                    print('guessed, correct : ', guessed[idx], correct[idx])
                    '''
                    if guessed[idx] == correct[idx]:
                        idx += 1
                        correctlyFound = True
                        while idx < len(guessed) and guessed[idx][0] == 'I': # Scan until it no longer starts with I.
                            if guessed[idx] != correct[idx]:
                                correctlyFound = False
                            idx += 1
                        if idx < len(guessed):
                            if correct[idx][0] == 'I': # The chunk in correct was longer.
                                correctlyFound = False
                        if correctlyFound:
                            correctCount += 1
                    else:
                        idx += 1
                else:  
                    idx += 1
        precision = 0
        if count > 0:    
            precision = float(correctCount) / count
        return precision

    @staticmethod
    def compute_f1(tag_preds, tag_corrects): 
        """Compute micro Fscore given tag-predictions and tag-corrects
        along with Precision, Recall.
        """
        prec = ChunkEval.compute_precision(tag_preds, tag_corrects)
        rec = ChunkEval.compute_precision(tag_corrects, tag_preds)
        f1 = 0
        if (rec+prec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec);
        return prec, rec, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    ev = ChunkEval()
    ev.eval()
