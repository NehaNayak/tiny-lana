cd /afs/cs.stanford.edu/u/nayakne/NLP-HOME/scr/tiny-lana/test

for trainDev in Train Dev
do
    filename='../params/os_EH_i100_vg_s'$trainDev'.txt'
    python findRank.py cosines/gloveRanks_EH_100.pickle $filename
done
