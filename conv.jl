using Pkg
Pkg.activate(".")
# Add the required packages to the environment
Pkg.add(["Images", "Flux", "MLDatasets", "CUDA", "Parameters", "ProgressMeter"])
using Base.Iterators: partition
using Flux
using Flux.Optimise: update!, train!
using Flux.Losses: logitbinarycrossentropy
using Images
using MLDatasets
using Statistics
using Printf
using Random
using CUDA
using ProgressMeter: @showprogress

Base.@kwdef struct HyperParams
    batch_size::Int = 128
    epochs::Int = 40
    verbose_freq::Int = 1000
    lr::Float32 = 0.0002
end

hparams = HyperParams()


dataset = MLDatasets.MNIST(:train)
dataset_test = MLDatasets.MNIST(:test)

images = dataset.features
labels = dataset.targets

indices = findall(val-> val < 2, labels)
images = images[:, :, indices]
labels = labels[indices]

indices = findall(val-> val < 2, dataset_test.targets)
images_test = dataset_test.features[:, :, indices]
labels_test = dataset_test.targets[indices]
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
dummy_input = reshape(images, 28, 28, 1, :)
dummy_output = convlayers(dummy_input[:, :, :, 1:1])

function EndLayers()
    osize = reduce(*, size(dummy_output)[1:3])
    return Chain(
        x->reshape(x, osize, :),
        Dense(osize, 1),
    )
end

# ConvModel = Chain(Convolutional(), EndLayers()) |> gpu
convmodel = lenet = Chain(
    Conv((5, 5), 1=>6, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 6=>16, relu),
    MaxPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, relu),
    Dense(120 => 84, relu), 
    Dense(84 => 10),
) |> gpu

function getloss(model, data, label)
    
    output = model(data)

    # one = logitbinarycrossentropy(output[findall(x->x==1, label)], 1)
    # zero = logitbinarycrossentropy(output[findall(x->x==0, label)], 0)


    return logitbinarycrossentropy(output[1], label)
end





function do_testing(test_data)
    # image_tensor = reshape(@.(2f0 * test_data - 1f0), 28, 28, 1, :)
    correct = 0
    total = 0
    for (data, label) in test_data
        # @info label
        # data = image_tensor[:, :, :, x:x]
        # label = labels_test[x:x]
        inference = convmodel(data) |> cpu
        for x in 1:size(inference, 2)
            # @info inference[1,x]
            pred = sigmoid(inference[1, x])
            if (round(pred) == label[x])
                correct += 1
            end
            total += 1

        end


        # data = reshape(@.(2f0* images_test[]))
    end
    
    @info "correct: $correct / $total"
end
function do_training()
    image_tensor = reshape(images, 28, 28, 1, :)
    test_image_tensor = reshape(images_test, 28, 28, 1, :)
    # labels = reshape(labels, 1, :)
    # label_tensor = reshape(labels, )

    data = [(image_tensor[:, :, :, r] |> gpu, labels[r]) for r in partition(1:size(image_tensor, 4), hparams.batch_size)]
    test_data = [(test_image_tensor[:, :, :, r] |> gpu, labels_test[r]) for r in partition(1:size(test_image_tensor, 4), hparams.batch_size)]
    opt = Flux.setup(Adam(hparams.lr), convmodel)

    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        train!(getloss, convmodel, data, opt)
        do_testing(test_data)
    end
end
do_training()
