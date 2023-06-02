using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Random: shuffle
using Base.Iterators: partition
using Flux
using Flux.MLUtils: chunk, batch, DataLoader
using Flux.Optimise: update!, train!
using Flux.Losses: logitcrossentropy
using Images
using MLDatasets
using Statistics
using Printf
using Random
using ProgressMeter: @showprogress
using JLD2
using FixedPointNumbers: N0f8
using OneHotArrays
using Images: load as imgload
using CUDA
@assert CUDA.functional(true)

datadir = "flower"
targetsize = (224, 224)

function crop_and_resize_image(image)
    if (size(image, 1) < size(image, 2))
        newx = size(image, 1)
        newy = size(image, 1)
    else
        newx = size(image, 2)
        newy = size(image, 2)
    end

    xlower = convert(Int16, floor(size(image, 1) / 2 - newx / 2)) + 1
    xupper = convert(Int16, floor(size(image, 1) / 2 + newx / 2 - 1)) + 1
    ylower = convert(Int16, floor(size(image, 2) / 2 - newy / 2)) + 1
    yupper = convert(Int16, floor(size(image, 2) / 2 + newy / 2 - 1)) + 1
    return imresize(image[xlower:xupper, ylower:yupper], targetsize)
end
is_too_small(image) = size(image, 1) < targetsize[1] || size(image, 2) < targetsize[2]

struct MyImageIterator
    classes::Vector{String}
    images::Vector{String}
    index::Int
end

get_image(class, image_name) = imgload(joinpath(datadir, class, image_name))

# function Base.getindex(iter::MyImageIterator, key...)
#     return Base.iterate()
# end

function Base.iterate(iter::MyImageIterator, state=1)
    if state > length(iter)
        return nothing
    end

    img = get_image(iter.classes[state], iter.images[state])
    # img = nothing
    # while state <= length(iter) && is_too_small(img)
    #     @info "Skipping state $state"
    #     state += 1
    #     if state > length(iter)
    #         return nothing
    #     end
    #     img = get_image(iter.classes[state], iter.images[state])
    # end
    cropped = crop_and_resize_image(img)
    # result = convert(Array{Float32, 3}, permutedims(channelview(cropped), (2, 3, 1)))
    result = permutedims(channelview(cropped), (2, 3, 1))
    
    # @info state, iter.classes[state]
    (result, iter.classes[state]), state+1
end


function Base.length(iter::MyImageIterator)
    length(iter.classes)
end

function images()
    classes = Vector{String}()
    images = Vector{String}()
    for class in cd(readdir, datadir)
        for image in cd(readdir, joinpath(datadir, class))
            classes = [classes; class]
            images = [images; image]
        end
    end

    MyImageIterator(classes, images, 1)
end

function gen_dataset() 
    # data = Array{Float16, 4}(undef, (128, 128, 3, 0))
    # labels = Array{UInt8, 2}(undef, (5, 0))
    # imgs = Vector{Tuple{Array{Float16, 3}, OneHotVector{UInt32}}}()
    datas = Array{Float32, 3}[]
    labels = OneHotVector{UInt32}[]


    @showprogress for img in images()
        data = img[1]
        lbl = img[2]
        label = Flux.onehot(lbl, sort(cd(readdir, datadir)))
        
        # push!(imgs, (data, label))
        push!(datas, data)
        push!(labels, label)
        # data = cat(data, onedata; dims=4)
        # labels = cat(labels, onehotlabel; dims=2)
    end
    # @save "preprocessed/images.jld2" imgs
    return batch(datas), batch(labels)
end

Base.@kwdef struct HyperParams
    batch_size::Int = 128
    epochs::Int = 100
    verbose_freq::Int = 1000
    lr::Float32 = 0.0002
end

function files_to_tensors(imgs, batchsize) 
    chunks = chunk(imgs;size=batchsize)
    return [batch(chunk) for chunk in chunks]

    # function get_tensors(chunk)
    #     # data, label = img

    #     datas = Array{Float16, 4}(undef, (targetsize..., 3, 0))
    #     labels = Array{UInt8, 2}(undef, (5, 0))
    #     for item in chunk 
    #         data, label = item
    #         datas = cat(datas, data; dims=4)
    #         labels = cat(labels, label; dims=2)
    #     end

    #     return datas, labels
    # end
    
    # [get_tensors(chunk) for chunk in chunks]
end

@info "Loading data..."
# @load "preprocessed/images.jld2" imgs
data, labels = gen_dataset()
data = Flux.normalise(data)

hparams = HyperParams()

indices = shuffle(1:size(data, 4))
train_count = convert(Int, round(size(data, 4) * 0.8))
train_indices = indices[1:train_count]
test_indices = indices[train_count+1:end]

train = data[:, :, :, train_indices]
@info "Train size: " * string(size(train))
test = data[:, :, :, test_indices]
@info "Test size: " * string(size(test))
train_labels = labels[:, train_indices]
test_labels = labels[:, test_indices]
train_dataloader = DataLoader((train, train_labels); batchsize=hparams.batch_size) |> gpu
test_dataloader = DataLoader((test, test_labels); batchsize=hparams.batch_size) |> gpu

# data = files_to_tensors(train, hparams.batch_size) test_data = only(files_to_tensors(test, length(dataset) - train_count))

randinit(shape...) = randn(Float32, shape...)
function Convolutional()
    return Chain(
        Conv((8, 8), 3 => 3*32, stride=4, pad=2, init=randinit),
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
dummy_output = convlayers(data[:, :, :, 1:2])

function EndLayers(osize)
    @info "Endlayers input size: $osize"
    return Chain(
        x->reshape(x, osize, :),
        # Dense(osize, 5*32),
        # Dense(5*32, 5*8),
        Dense(osize, 5),
    )
end
convmodel = Chain(Convolutional(), EndLayers(reduce(*, size(dummy_output)[1:3]))) |> gpu



function do_training()
    @info "Starting training"
    opt = Flux.setup(Adam(hparams.lr), convmodel)
    function mycb()
        loss = 0
        correct = 0
        incorrect = 0
        for (x,y) in test_dataloader
            ŷ = convmodel(x)
            # @info ŷ
            # @info y
            loss += Flux.logitcrossentropy(ŷ, y)  # did not include softmax in the model
            correct += count(Flux.onecold(ŷ) .== Flux.onecold(y))
            incorrect += count(Flux.onecold(ŷ) .!= Flux.onecold(y))
        end
        @info "Gross loss: $loss"
        loss = loss / length(test_dataloader)
        @info "Gross correct: $correct; gross incorrect: $incorrect"
        acc = 100 * correct / (correct + incorrect)
        @info "Loss: $loss; accuracy: $acc"
        @save "runs/convmodel.jld2" convmodel
    end
    cb = Flux.throttle(mycb, 10)
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        @showprogress for (x,y) in train_dataloader
            grads = gradient(m -> logitcrossentropy(m(x), y), convmodel)[1]
            update!(opt, convmodel, grads)
        end
        cb()
    end
end
do_training()
