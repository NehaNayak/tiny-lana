import random
import sys

Hyper = []
NotHyper = []

for line in sys.stdin:
    (w1, rel, w2) = line.split()
    w1 = "_".join(w1.split('_')[2:-1])
    w2 = "_".join(w2.split('_')[2:-1])
    if rel=="_type_of":
        Hyper.append((w1, w2))
    else:
        NotHyper.append((w1, w2))
    
NotHyper = random.sample(NotHyper, len(Hyper))

HyperLabelled = map(lambda x : (x[0], x[1], "1"), Hyper)
NotHyperLabelled = map(lambda x : (x[0], x[1], "0"), NotHyper)

BothLabelled = HyperLabelled+NotHyperLabelled
random.shuffle(BothLabelled)

for pair in BothLabelled:
    sys.stdout.write("\t".join(list(pair))+"\n")
