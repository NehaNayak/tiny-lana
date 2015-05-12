require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

function readSet(path)
    local m = torch.randn(cmdparams.inputSize):zero()
    local f =assert(io.open(path, "r"))

    local set_in = nil
    local set_out = nil

    while true do
        line = f:read()
        if not line then break end
        for hypo,hyper in string.gmatch(line, "(%S+)%s(%S+)") do
            local vhypo = emb_vecs[emb_vocab:index(hypo)]
            local vhyper = emb_vecs[emb_vocab:index(hyper)]
            if vhypo~= nil and vhyper~=nil then

                vin = vhypo:typeAs(m)
                vout = vhyper:typeAs(m)

                if set_in ==nil then
                    set_in = vin:clone()
                    set_out= vout:clone()
                else
                    set_in = torch.cat(set_in,vin,2)
                    set_out = torch.cat(set_out,vout,2)
                end
            end
        end
    end

    return {set_in:t(),set_out:t()}
   
end

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')
cmd:option('-hiddenSize',500,'size of hidden layer')
cmd:option('-learningRate',0.05,'learning rate')
cmd:option('-regCoeff',0.001,'regularisation coefficient')
cmd:option('-endLimit',100,'maximum number of iterations with decreasing dev loss')
cmd:option('-pairSet','EH','which relation')
cmd:option('-vecSet','g.direct','glove vectors = \'g~~\'; word2vec vectors = \'v~~\'')

cmd:option('-printFreq',1,'number of iterations after which to print loss')
cmdparams = cmd:parse(arg)

-- Create Train and Dev sets 

trainPath = table.concat({
        '../pairs/pairs', 
        cmdparams.pairSet,
        '_Train.txt'
        },"")

train = readSet(trainPath) 
train_in=train[1]
train_out=train[2]

devPath = table.concat({
        '../pairs/pairs', 
        cmdparams.pairSet,
        '_Dev.txt'
        },"")

dev = readSet(devPath) 
dev_in=dev[1]
dev_out=dev[2]


--emb_dir = '/scr/kst/data/wordvecs/glove/'
emb_dir = '/Users/neha/wordvecs/glove/'
emb_prefix = emb_dir .. 'glove.6B'
emb_vocab, emb_vecs = torchnlp.read_embedding(
emb_prefix .. '.vocab',
emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')



-- Define model

model = nn.Sequential()
model:add(nn.Linear(cmdparams.inputSize, cmdparams.hiddenSize)) 
model:add(nn.Tanh())
model:add(nn.Linear(cmdparams.hiddenSize, cmdparams.inputSize))
model:add(nn.Tanh())

criterion = nn.CosineEmbeddingCriterion()

-- Train

x, dl_dx = model:getParameters()

-- Define closure

feval = function()
   i = (i or 0) + 1
   if i > (#train_in)[1] then i = 1 end

   local input = train_in[i]:clone()
   local target = train_out[i]:clone()

   dl_dx:zero()

   local loss_x = criterion:forward({model:forward(input), target},1)
   loss_x = loss_x + cmdparams.regCoeff*torch.norm(x,2)^2/2
   model:backward(input, criterion:backward({model.output, target},1)[1])
   dl_dx:add(x:clone():mul(cmdparams.regCoeff))

   return loss_x, dl_dx

end

-- SGD

sgd_params = {
   learningRate = cmdparams.learningRate,
}

local prev_dev_loss=math.huge
local dev_loss=math.huge
bestLoss=math.huge
lessCount=0

for i = 1, 10000 do

    train_loss = 0
    prev_model=model:clone()
  
    for j = 1,(#train_in)[1] do
        _,fs = optim.adagrad(feval,x,sgd_params)
        train_loss = train_loss + fs[1]
    end
    train_loss = train_loss / (#train_in)[1]
    
    prev_dev_loss=dev_loss
    dev_loss=0

    x, dl_dx = model:getParameters()
    for j = 1,(#dev_in)[1] do
        local loss = criterion:forward({model:forward(dev_in[j]), dev_out[j]},1)
        loss = loss + cmdparams.regCoeff*torch.norm(x,2)^2/2
        dev_loss = dev_loss+loss
    end
    print( cmdparams.regCoeff*torch.norm(x,2)^2/2)
    dev_loss = dev_loss / (#dev_in)[1]
    
    if dev_loss>prev_dev_loss then
        if prev_dev_loss<bestLoss then
            bestLoss = prev_dev_loss
            bestModel=prev_model:clone()
            torch.save(tempOutPath, bestModel)
            print("best model found at " .. lessCount)
        end
        lessCount=lessCount+1
        if lessCount==cmdparams.endLimit then
            torch.save(outPath, bestModel)
            break
        end
    end

    if i%cmdparams.printFreq==0 then
        print('current train loss = ' .. train_loss)
        print('current dev loss = ' .. dev_loss)
    end
end
