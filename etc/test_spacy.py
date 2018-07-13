from __future__ import print_function
import sys
import argparse

def get_entity(doc, begin, end):
    for ent in doc.ents:
        # check included
        if ent.start_char <= begin and end <= ent.end_char:
            if ent.start_char == begin: return 'B-' + ent.label_
            else: return 'I-' + ent.label_
    return 'O'
     
def build_bucket(nlp, line):
    bucket = []
    uline = line.decode('utf-8','ignore') # unicode
    doc = nlp(uline)
    print('postag:')
    seq = 0
    for token in doc:
        begin = token.idx
        end   = begin + len(token.text) - 1
        temp = []
        print(token.i, token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop, begin, end)
        temp.append(token.text)
        temp.append(token.tag_)
        temp.append('O')     # no chunking info
        entity = get_entity(doc, begin, end)
        temp.append(entity)  # label
        utemp = ' '.join(temp)
        bucket.append(utemp.encode('utf-8'))
        seq += 1
    print('')
    print('named entity:')
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
    print('')
    print('noun chunk:')
    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_,chunk.root.head.text)
    print('')
    return bucket

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    import spacy
    nlp = spacy.load('en')

    while 1:
        try: line = sys.stdin.readline()
        except KeyboardInterrupt: break
        if not line: break
        line = line.strip()
        if not line: continue
        # Create bucket
        try: bucket = build_bucket(nlp, line)
        except Exception as e:
            sys.stderr.write(str(e) +'\n')
            continue
        for i in range(len(bucket)):
            out = bucket[i]
            sys.stdout.write(out + '\n')
        sys.stdout.write('\n')

