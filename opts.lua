"""
Parse arguments for the model to train.
"""
local M = {}

function M.parse(arg)
    local cmd = torch.CmdLine();
    cmd:text()
    cmd:text('The German Traffic Sign Recognition Benchmark: A multi-class classification ')
    cmd:text()
    cmd:text('Options:')
    cmd:option('-data',             '',             'Path to dataset')
    cmd:option('-val',              10,             'Percentage to use for validation set')
    cmd:option('-nEpochs',          300,            'Maximum epochs')
    cmd:option('-batchsize',        128,            'Batch size for epochs')
    cmd:option('-nThreads',         5,              'Number of dataloading threads')
    cmd:option('-manualSeed',       '0',            'Manual seed for RNG')
    cmd:option('-LR',               0.1,            'initial learning rate')
    cmd:option('-momentum',         0.9,            'momentum')
    cmd:option('-weightDecay',      1e-4,           'weight decay')
    cmd:option('-logDir',           'logs',         'log directory')
    cmd:option('-model',            '',             'Model to use for training')
    cmd:option('-cudnn',            'false',             'Use cuda tensor')
    cmd:option('-balance',          'false',             'Rebalance classes')
    cmd:option('-angle',            '10',             'Rotate image randomly between - x and x degree')
    cmd:option('-image',            '32',             'Rotate image randomly between - x and x degree')

    local opt = cmd:parse(arg or {})

    if opt.model == '' or not paths.filep('models/'..opt.model..'.lua') then
        cmd:error('Invalid model ' .. opt.model)
    end

    return opt
end

return M
