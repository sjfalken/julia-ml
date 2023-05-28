using Pkg
Pkg.activate(".")
# Add the required packages to the environment
# Pkg.add(["Images", "Flux", "MLDatasets", "CUDA", "Parameters", "ProgressMeter"])
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

targetsize = (128, 128)


hparams = HyperParams()
# train_dataset = MLDatasets.MNIST(:train)
# test_dataset = MLDatasets.MNIST(:test)
# include("data.jl")

filenames = cd(readdir, "preprocessed")

# it = images()
filenames = shuffle(filenames)
# train_count = convert(Int, round(length(filenames) * 0.1))
train_count = convert(Int, round(length(filenames) * 0.05))

train_files = filenames[1:train_count]
test_files = filenames[train_count+1:end]


# train_data = images[1:4000]
# test_data = images[4001:5000]


# @enum DataSplit test train

# train_loader = Flux.DataLoader(())


randinit(shape...) = randn(Float32, shape...)
function Convolutional()
    return Chain(
        Conv((4, 4), 3 => 3*32, stride=2, pad=1, init=randinit, groups=3),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
        Conv((4, 4), 3*32 => 3*64, stride=2, pad=1, init=randinit, groups=3),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
    )
end
convlayers = Convolutional()
# dummy_input = reshape(train_dataset.features, 28, 28, 1, :)
dummyfile = test_files[1]
# dummydata = Array{N0f8, 3}(undef, (128, 128, 3))
# dummylabel = Array{UInt32, 1}(undef, (5,))
@load "preprocessed/$dummyfile" data onehotlabel
dummy_output = convlayers(reshape(data, 128, 128, 3, 1))

function EndLayers()
    osize = reduce(*, size(dummy_output)[1:3])
    return Chain(
        x->reshape(x, osize, :),
        Dense(osize, 10),
    )
end

convmodel = Chain(Convolutional(), EndLayers()) |> gpu
# convmodel = lenet = Chain(
#     Conv((5, 5), 1=>6, relu),
#     MaxPool((2, 2)),
#     Conv((5, 5), 6=>16, relu),
#     MaxPool((2, 2)),
#     Flux.flatten,
#     Dense(256 => 120, relu),
#     Dense(120 => 84, relu), 
#     Dense(84 => 10),
# ) |> gpu

function getloss(model, data, label)
    
    output = model(data)

    # one = logitbinarycrossentropy(output[findall(x->x==1, label)], 1)
    # zero = logitbinarycrossentropy(output[findall(x->x==0, label)], 0)


    return logitcrossentropy(output, label)
end


function files_to_tensors(files) 
    # file_chunks = chunk(files;size=batchsize)

    function get_tensors(files)

        resultdata = Array{N0f8, 4}(undef, (128, 128, 3, 0))
        labels = Array{Int, 2}(undef, (5, 0))
        for file in files
            @load "preprocessed/$file" data onehotlabel
            resultdata = cat(resultdata, data; dims=4)
            labels = cat(labels, onehotlabel; dims=2)
        end

        return resultdata, labels
    end
    
    get_tensors(files)
end


function do_training()
    opt = Flux.setup(Adam(hparams.lr), convmodel)
    # train_tensor = reduce(cat, [])
    # @info size(train_tensor)
    # filename_chunks = chunk(train_files; size=hparams.batch_size)
    loader = Flux.DataLoader(files_to_tensors(train_files); batchsize=hparams.batch_size) |> gpu
    function mycb()
        # (x,y) = only(loader(test_dataset; batchsize=length(test_dataset)))  # make one big batch
        # ŷ = convmodel(x)
        # loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
        # acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
        # # (; loss, acc, split=data.split)  # return a NamedTuple
        # data_batches = files_to_batches(train_files, hparams.batch_size)
        # @info "Loss: $loss; accuracy: $acc" @save "runs/convmodel.jld2" convmodel
    end
    cb = Flux.throttle(mycb, 10)
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        @showprogress for d in loader
            ∂L∂m = gradient(getloss, convmodel, d...)[1]
            update!(opt, convmodel, ∂L∂m)
        end
        cb()
        # train!(getloss, convmodel, loader(train_dataset), opt, cb = Flux.throttle(mycb, 5))
        # train!
        # do_testing(test_data)
    end
end
do_training()
