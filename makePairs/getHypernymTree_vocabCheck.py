import nltk
import sys
from nltk.corpus import wordnet as wn

def printChildren(currSynset):
    if len(currSynset.hyponyms())==0:
        return
    else:
        for hyponym in currSynset.hyponyms():
            hyper = currSynset.lemmas()[0].name()
            hypo = hyponym.lemmas()[0].name()
            if hypo in Vocab and hyper in Vocab:
                sys.stdout.write(\
                hyponym.lemmas()[0].name()+\
                "\t"+\
                currSynset.lemmas()[0].name()+\
                "\n")
        for hyponym in currSynset.hyponyms():
            printChildren(hyponym)

def main():
    global Vocab
    Vocab = set()
    for line in sys.stdin:
        Vocab.add(line[:-1])
        
    organism = wn.synset(sys.argv[1])
    printChildren(organism)
    

if __name__ =="__main__":
    main()
    
