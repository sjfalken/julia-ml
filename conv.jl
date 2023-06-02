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
using ColorTypes: RGB
using MLDatasets
using Statistics
using Printf
using Random
using ProgressMeter: @showprogress
using JLD2
using FixedPointNumbers: N0f8
using OneHotArrays
using Images: load as imgload, save as imgsave
using CUDA
@assert CUDA.functional(true)

datadir = "flower"
targetsize = (300, 300)
# num_images = 5000
num_images = 5000

if !any("runs" .== readdir("."))
    mkdir("runs")
end

if !any("output" .== readdir("."))
    mkdir("output")
end

if !any("flower" .== readdir("."))
    @warn "flower dataset missing"
end

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

function get_img_from_data(data)
    @assert all(size(data) .== (targetsize..., 3))

    data = convert(Array{N0f8}, clamp.(data .+ 0.5, 0, 1))
    data = permutedims(data, (3, 1, 2))

    # imgsave(path, colorview(RGB, data))
    return colorview(RGB, data)
    
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


    count = 0
    @showprogress for img in images()
        if count >= num_images
            break
        end
        count += 1
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
    batch_size::Int = 32
    epochs::Int = 300
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
data = data .- 0.5f0 # normalize

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
        Conv((8, 8), 3 => 3*16, stride=4, pad=2, init=randinit),
        MaxPool((4, 4)),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
        Conv((2, 2), 3*16 => 3*32, stride=1, pad=2, init=randinit),
        MaxPool((4, 4)),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
        Conv((2, 2), 3*32 => 3*64, stride=1, pad=1, init=randinit),
        MaxPool((4, 4)),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.2),
    )
end
convlayers = Convolutional()
dummy_output = convlayers(data[:, :, :, 1:2])

# save_example_img("example.png", data[:, :, :, 1])

function EndLayers(osize)
    @info "Endlayers input size: $osize"
    return Chain(
        x->reshape(x, osize, :),
        # Dense(osize, 5*32),
        # Dense(5*32, 5*8),
        Dense(osize, 5),
    )
end
# runs = cd(readdir, "runs")
# idx = findfirst("convmodel.jld2" .== runs)
# if idx !== nothing
#     @info "starting with existing model"
#     @load "runs/convmodel.jld2" convmodel
#     convmodel = convmodel |> gpu
# else
#     @info "creating model"
#     convmodel = Chain(Convolutional(), EndLayers(reduce(*, size(dummy_output)[1:3]))) |> gpu
# end
ConvModel() = Chain(Convolutional(), EndLayers(reduce(*, size(dummy_output)[1:3])))
convmodel = ConvModel() |> gpu


struct ModelState
    model
    loss::Float32
end


function do_training()
    @info "Starting training"
    opt = Flux.setup(Adam(hparams.lr), convmodel)
    function mycb(epoch)
        loss = 0
        correct = 0
        incorrect = 0
        img_batch_idx_loss = Tuple{Int, Int, Float32}[]

        # loadercpu = test_dataloader |> cpu

        
        for (j, (x,y)) in enumerate(test_dataloader)
            ŷ = convmodel(x)

            yhatcpu = ŷ |> cpu
            ycpu = y |> cpu


            for i in axes(ycpu, 2)

                l = Flux.logitcrossentropy(yhatcpu[:, i], ycpu[:, i])  # did not include softmax in the model
                push!(img_batch_idx_loss, (j, i, l))
                loss += l
                if Flux.onecold(yhatcpu[:, i]) == Flux.onecold(ycpu[:, i])
                    correct += 1
                else
                    incorrect += 1
                end
            end
        end

        sort!(img_batch_idx_loss; lt = (a, b) -> a[3] < b[3], rev=true)

        toplossimages = map(t -> test[:, :, :, (t[1] - 1)*hparams.batch_size + t[2]], img_batch_idx_loss[1:25])
        imgbatch = batch([get_img_from_data(img) for img in toplossimages])
        m = mosaicview(imgbatch; ncol = 5)
        imgsave("output/$epoch.png", m)

        @info "Gross loss: $loss"
        loss = loss / (correct+incorrect)
        @info "Gross correct: $correct; gross incorrect: $incorrect"
        acc = 100 * correct / (correct + incorrect)
        @info "average loss: $loss; accuracy: $acc"

        state = ModelState(Flux.state(convmodel), loss)
        if ispath("runs/convmodel.jld2")
            jldopen("runs/convmodel.jld2", "r") do file
                if file["modelstate"].loss < state.loss
                    state = file["modelstate"]
                end
            end
        end

        jldsave("runs/convmodel.jld2"; modelstate=state)
        # @save "runs/convmodel_$epoch.jld2" convmodel |> cpu
    end
    cb = Flux.throttle(mycb, 10)
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        @showprogress for (x,y) in train_dataloader
            grads = gradient(m -> logitcrossentropy(m(x), y), convmodel)[1]
            update!(opt, convmodel, grads)
        end
        cb(ep)
    end
end
do_training()
