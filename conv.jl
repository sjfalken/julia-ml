using Pkg
Pkg.activate(".")
Pkg.add(["Images", "Flux", "MLDatasets", "CUDA", "Parameters", "ProgressMeter"])
using Random: shuffle
using Base.Iterators: partition
using Flux
using Flux.MLUtils: chunk
using Flux.Optimise: update!, train!
using Flux.Losses: logitcrossentropy
using Images
using MLDatasets
using Statistics
using Printf
using Random
using CUDA
using FixedPointNumbers: N0f8
using ProgressMeter: @showprogress
using JLD2
Base.@kwdef struct HyperParams
    batch_size::Int = 128
    epochs::Int = 40
    verbose_freq::Int = 1000
    lr::Float32 = 0.0002
end

targetsize = (256, 256)

function files_to_tensors(imgs, batchsize) 
    chunks = chunk(imgs;size=batchsize)

    function get_tensors(chunk)
        # data, label = img

        datas = Array{Float16, 4}(undef, (targetsize..., 3, 0))
        labels = Array{UInt8, 2}(undef, (5, 0))
        for item in chunk 
            data, label = item
            datas = cat(datas, data; dims=4)
            labels = cat(labels, label; dims=2)
        end

        return datas, labels
    end
    
    [get_tensors(chunk) |> gpu for chunk in chunks]
end

@load "preprocessed/images.jld2" imgs

hparams = HyperParams()



train_count = convert(Int, round(length(imgs) * 0.8))

train = imgs[1:train_count]
test = imgs[train_count+1:end]

data = files_to_tensors(train, hparams.batch_size)
test_data = only(files_to_tensors(test, length(imgs) - train_count))

randinit(shape...) = randn(Float32, shape...)
function Convolutional()
    return Chain(
        Conv((4, 4), 3 => 3*32, stride=2, pad=1, init=randinit),
        MaxPool((4, 4)),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
        Conv((4, 4), 3*32 => 3*64, stride=2, pad=1, init=randinit),
        MaxPool((4, 4)),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
    )
end
convlayers = Convolutional()
dummy_output = convlayers(reshape(imgs[1][1], targetsize..., 3, 1))

function EndLayers(osize)
    # @info osize
    return Chain(
        x->reshape(x, osize, :),
        Dense(osize, 5*32),
        Dense(5*32, 5*8),
        Dense(5*8, 5),
    )
end
convmodel = Chain(Convolutional(), EndLayers(reduce(*, size(dummy_output)[1:3]))) |> gpu



function do_training()
    @info "Starting training"
    opt = Flux.setup(Adam(hparams.lr), convmodel)
    function mycb()
        (x,y) = test_data
        ŷ = convmodel(x)
        loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
        acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
        @info "Loss: $loss; accuracy: $acc"
        @save "runs/convmodel.jld2" convmodel
    end
    cb = Flux.throttle(mycb, 10)
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        @showprogress for (x,y) in data
            grads = gradient(m -> logitcrossentropy(m(x), y), convmodel)[1]
            update!(opt, convmodel, grads)
        end
        cb()
    end
end
do_training()
