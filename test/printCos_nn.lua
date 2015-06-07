require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

function cosine(v, w)
    return v:dot(w) / v:norm() / w:norm()
end
-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')
cmd:option('-pairSet','EH','which relation')
cmd:option('-trainDev','Train','which set')
cmd:option('-vecSet','g','glove vectors = \'g~~\'; word2vec vectors = \'v~~\'')

cmdparams = cmd:parse(arg)

-- Load word embeddings

emb_dir = '/scr/nayakne/wordvecs/glove/'
emb_prefix = emb_dir .. 'glove.6B'
emb_vocab, emb_vecs = torchnlp.read_embedding(
emb_prefix .. '.vocab',
emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')

-- Create Train and Dev sets 

setPath = table.concat({
        '../pairs/pairs', 
        cmdparams.pairSet,
        '_NN_',
        cmdparams.trainDev,
        '.txt'
        },"")

outputPath = table.concat({
        '../params/nn_',
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
       '_v',
        cmdparams.vecSet,
        '_s',
        cmdparams.trainDev,
        '.txt'
        },"")
    
local outFile =assert(io.open(outputPath, "w"))

local m = torch.randn(cmdparams.inputSize):zero()

-- Define model
local f = assert(io.open(setPath, "r"))
while true do

    line = f:read()
    if not line then break end

    for win,wout,wnn in string.gmatch(line, "(%S+)%s(%S+)%s(%S+)") do

        local vin = emb_vecs[emb_vocab:index(win)]:typeAs(m)
        local vout = emb_vecs[emb_vocab:index(wout)]:typeAs(m)
        local vnn = emb_vecs[emb_vocab:index(wnn)]:typeAs(m)

        predictedCosine = cosine(vnn,vout)
        selfCosine = cosine(vin, vout)

        outputLine = table.concat({
        win,
        wout,
        predictedCosine,
        selfCosine
        },"\t")

        outFile:write(outputLine .. "\n")

    end
end
