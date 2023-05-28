using Pkg
Pkg.activate(".")
# Add the required packages to the environment
# Pkg.add(["Images", "Flux", "MLDatasets", "CUDA", "Parameters", "ProgressMeter"])
using Base.Iterators: partition
using Flux
using Flux.Optimise: update!, train!
using Flux.Losses: logitcrossentropy
using Images
using MLDatasets
using Statistics
using Printf
using Random
using CUDA
using ProgressMeter: @showprogress
using JLD2
Base.@kwdef struct HyperParams
    batch_size::Int = 128
    epochs::Int = 40
    verbose_freq::Int = 1000
    lr::Float32 = 0.0002
end

hparams = HyperParams()

train_dataset = MLDatasets.MNIST(:train)
test_dataset = MLDatasets.MNIST(:test)

function loader(data::MNIST; batchsize=hparams.batch_size)
    x4dim = reshape(data.features, 28,28,1,:)   # insert trivial channel dim
    yhot = Flux.onehotbatch(data.targets, 0:9)  # make a 10×60000 OneHotMatrix
    Flux.DataLoader((x4dim, yhot); batchsize=batchsize, shuffle=true) |> gpu
    # Dataloader()
end

randinit(shape...) = randn(Float32, shape...)
function Convolutional()
    return Chain(
        Conv((1, 4), 1 => 64, stride=2, pad=1, init=randinit),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
        Conv((4, 4), 64 => 128, stride=2, pad=1, init=randinit),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
    )
end
convlayers = Convolutional()
dummy_input = reshape(train_dataset.features, 28, 28, 1, :)
dummy_output = convlayers(dummy_input[:, :, :, 1:1])

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




function do_training()
    # image_tensor = reshape(images, 28, 28, 1, :)
    # test_image_tensor = reshape(images_test, 28, 28, 1, :)
    # labels = reshape(labels, 1, :)
    # label_tensor = reshape(labels, )

    # data = [(image_tensor[:, :, :, r] |> gpu, labels[r]) for r in partition(1:size(image_tensor, 4), hparams.batch_size)]
    # test_data = [(test_image_tensor[:, :, :, r] |> gpu, labels_test[r]) for r in partition(1:size(test_image_tensor, 4), hparams.batch_size)]
    opt = Flux.setup(Adam(hparams.lr), convmodel)

    function mycb()
        (x,y) = only(loader(test_dataset; batchsize=length(test_dataset)))  # make one big batch
        ŷ = convmodel(x)
        loss = Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
        acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
        # (; loss, acc, split=data.split)  # return a NamedTuple

        @info "Loss: $loss; accuracy: $acc"
    end
    cb = Flux.throttle(mycb, 10)
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        @showprogress for d in loader(train_dataset)
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
