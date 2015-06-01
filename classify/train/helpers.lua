getEmbeddings = function()
    emb_dir = '/scr/kst/data/wordvecs/glove/'
    emb_prefix = emb_dir .. 'glove.6B'
    emb_vocab, emb_vecs = torchnlp.read_embedding(
    emb_prefix .. '.vocab',
    emb_prefix .. '.' .. cmdparams.inputSize ..'d.th')
    return emb_vocab, emb_vecs
end

function readSetCat(size,path)
    local m = torch.randn(size):zero()
    local f =assert(io.open(path, "r"))

    local pairList = {}
    local set_in = nil
    local set_out = nil

    while true do
        line = f:read()
        if not line then break end
        for hypo,hyper,label in string.gmatch(line, "(%S+)%s(%S+)%s(%S)") do
            vhypo = m:clone()
            for subWord in string.gmatch(hypo,"([^_]+)") do
                if pcall(function () emb_vocab:index(subWord) end) then
                vhypo:add(emb_vecs[emb_vocab:index(subWord)]:typeAs(m)) end
            end
            vhyper = m:clone()
            for subWord in string.gmatch(hyper,"([^_]+)") do
                if pcall(function () emb_vocab:index(subWord) end) then
                vhyper:add(emb_vecs[emb_vocab:index(subWord)]:typeAs(m)) end
            end

            if vhypo~= nil and vhyper~=nil then

                table.insert(pairList, hypo .. "\t" .. hyper)
                vin = torch.cat(vhyper:typeAs(m),vhypo:typeAs(m))

                if label=="0" then vout = torch.Tensor({1,0})
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
        --model:add(nn.LogSoftMax())
	elseif whichModel=='1l' then
		model:add(nn.Linear(2*cmdparams.inputSize, 2*cmdparams.inputSize*cmdparams.hiddenFactor))
		model:add(nn.Tanh())
		model:add(nn.Linear(2*cmdparams.inputSize*cmdparams.hiddenFactor,2))
        model:add(nn.LogSoftMax())
	elseif whichModel=='2l' then
		model:add(nn.Linear(2*cmdparams.inputSize, 2*cmdparams.inputSize*cmdparams.hiddenFactor))
		model:add(nn.Tanh())
		model:add(nn.Linear(2*cmdparams.inputSize*cmdparams.hiddenFactor, 2*cmdparams.inputSize*cmdparams.hiddenFactor))
		model:add(nn.Tanh())
		model:add(nn.Linear(2*cmdparams.inputSize*cmdparams.hiddenFactor,2))
        model:add(nn.LogSoftMax())
	end
    return model
end

function setError(model,set_in,set_out)
    correct = 0
    wrong = 0
     for j = 1,(#set_in)[1] do
        local label = set_out[j][1]+2*set_out[j][2]
        local pred = model:forward(set_in[j])
        if pred[1]>pred[2] then res = 1 else res = 2 end
        if res==label then correct = correct + 1 else wrong = wrong + 1 end
    end
    return wrong/(correct+wrong)      
end
