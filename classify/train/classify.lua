require 'torch'
require 'optim'
require 'nn'
require 'torchnlp'

-- Command line arguments

cmd = torch.CmdLine()

cmd:option('-inputSize',100,'size of input layer')
cmd:option('-learningRate',0.5,'learning rate')
cmd:option('-endLimit',10,'maximum number of iterations with decreasing dev loss')
cmd:option('-pairSet','BH','which relation')
cmd:option('-vecSet','g.direct','glove vectors = \'g~~\'; word2vec vectors = \'v~~\'')
cmd:option('-printFreq',10e4,'number of iterations after which to print loss')
cmd:option('-model','af', which model to use)
cmdparams = cmd:parse(arg)

-- Load word embeddings

emb_vocab, emb_vecs = getEmbeddings()

-- Create Train and Dev sets 

trainPath = '../pairs/pairs' .. cmdparams.pairSet .. '_Train.txt'
train = readSet(cmdparams.inputSize, trainPath) 
train_in=train[1] ; train_out=train[2] ; trainPairs = train[3]

devPath = '../pairs/pairs' .. cmdparams.pairSet .. '_Dev.txt'
dev = readSet(cmdparams.inputSize, devPath) 
dev_in=dev[1] ; dev_out=dev[2] ; devPairs=dev[3]

outPath = table.concat({
        '../params/afC_', 
        cmdparams.pairSet,
        '_i',
        cmdparams.inputSize,
        '_lr',
        cmdparams.learningRate,
        '_el',
        cmdparams.endLimit,
        '_v',
        cmdparams.vecSet,
        '.th'
        },"")

-- Define model

model = makeModel(cmdParams.model)
criterion = ClassNLLCriterion

-- Train

x, dl_dx = model:getParameters()

-- Define closure

feval = function()
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#train_in)[1] then _nidx_ = 1 end

   local input = train_in[_nidx_]:clone()
   local target = train_out[_nidx_]:clone()

   dl_dx:zero()

    local label = target[1]+2*target[2]
    local loss_x = criterion:forward(model:forward(input), label)
    model:backward(input, criterion:backward(model.output, label))

   return loss_x, dl_dx

end

-- SGD

sgd_params = {
   learningRate = cmdparams.learningRate,
   learningRateDecay = 0,
   weightDecay = 0,
   momentum = 0
}

local prev_dev_loss=math.huge
local dev_loss=math.huge
bestLoss=math.huge
lessCount=0

for i = 1,10000 do

    train_loss = 0
    prev_model=model:clone()

    for j = 1,(#train_in)[1] do
        _,fs = optim.adagrad(feval,x,sgd_params)
        train_loss = train_loss + fs[1]
    end
    train_loss = train_loss / (#train_in)[1]
    
    prev_dev_loss=dev_loss
    dev_loss=0

    for j = 1,(#dev_in)[1] do
        local label = dev_out[j][1]+2*dev_out[j][2]
        local loss = criterion:forward(model:forward(dev_in[j]), label)
        dev_loss = dev_loss+loss
    end
    dev_loss = dev_loss / (#dev_in)[1]
    
    if dev_loss>prev_dev_loss or (prev_dev_loss - dev_loss) < 10e-6 then
        if prev_dev_loss<bestLoss then
            bestLoss = prev_dev_loss
            bestModel=prev_model:clone()
            print("best model found at " .. lessCount)
        end
        lessCount=lessCount+1
        if lessCount==cmdparams.endLimit then
            torch.save(outPath, bestModel)
            --for k = 1,(#dev_in)[1] do
                --output = bestModel:forward(dev_in[k])
                --if output[1]>output[2] then res = 0 else res = 1 end
                --local label = dev_out[k][1]+2*dev_out[k][2]-1
                --print(devPairs[k] .. "\t" .. label .. "\t" .. res)
            --end
            break
        end
    end

    if i%cmdparams.printFreq==0 then
        print('current train loss = ' .. train_loss)
        print('current dev loss = ' .. dev_loss)
    end
end

function getEmbeddings()
    emb_dir = '/scr/kst/data/wordvecs/glove/'
    emb_prefix = emb_dir .. 'glove.6B'
    emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')
    return emb_vocab, emb_vecs
end

function readSet(size,path)
    local m = torch.randn(size):zero()
    local f =assert(io.open(path, "r"))

    local pairList = {}
    local set_in = nil
    local set_out = nil

    while true do
        line = f:read()
        if not line then break end
        for hypo,hyper,label in string.gmatch(line, "(%S+)%s(%S+)%s(%S)") do
            local vhypo = emb_vecs[emb_vocab:index(hypo)]
            local vhyper = emb_vecs[emb_vocab:index(hyper)]
            if vhypo~= nil and vhyper~=nil then

                table.insert(pairList, hypo .. "\t" .. hyper)
                vin = torch.cat(vhyper:typeAs(m),vhypo:typeAs(m))

                if label=='0' then vout = torch.Tensor({1,0})
                else vout = torch.Tensor({0,1}) end

                if set_in ==nil then
                    set_in = vin:clone()/vin:norm()
                    set_out= vout:clone()
                else
                    set_in = torch.cat(set_in,vin/vin:norm(),2)
                    set_out = torch.cat(set_out,vout,2)
                end
            end
        end
    end

    return {set_in:t(),set_out:t(), pairList}
   
end

function makeModel(whichModel)
	model = nn.Sequential()
	if whichModel=='af' then
		model:add(nn.Linear(2*cmdparams.inputSize, 2))
	else if whichModel=='1l' then
		model:add(nn.Linear(2*cmdparams.inputSize, 2*cmdparams.inputSize*cmdparams.hiddenFactor))
		model.add(nn.Tanh)
		model:add(nn.Linear(2*cmdparams.inputSize*cmdparams.hiddenFactor,2))
	else if whichModel=='2l' then
		model:add(nn.Linear(2*cmdparams.inputSize, 2*cmdparams.inputSize*cmdparams.hiddenFactor))
		model.add(nn.Tanh)
		model:add(nn.Linear(2*cmdparams.inputSize*cmdparams.hiddenFactor, 2*cmdparams.inputSize*cmdparams.hiddenFactor))
		model.add(nn.Tanh)
		model:add(nn.Linear(2*cmdparams.inputSize*cmdparams.hiddenFactor,2))
	end
end

