using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Random: shuffle
using Base.Iterators: partition
using Flux
using Flux.MLUtils: chunk, batch, DataLoader
using Flux.Optimise: update!, train!
using Flux.Losses: logitcrossentropy
using Images, CoordinateTransformations, Rotations
using ColorTypes 
using MLDatasets
using Statistics
using Printf
using Random
using ProgressMeter: @showprogress
using JLD2
using FixedPointNumbers: N0f8
using OneHotArrays
using Images: load as imgload, save as imgsave
using TensorBoardLogger, Logging
using CUDA
using Dates
@assert CUDA.functional(true)

Base.@kwdef struct HyperParams
    batch_size::Int = 32
    epochs::Int = 300
    verbose_freq::Int = 1000
    lr::Float32 = 0.0002
end

datadir = "flower"
runsdir = "runs"
targetsize = (200, 200)
num_images = 5000


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

# get_image(class, image_name) = imgload(joinpath(datadir, class, image_name))

function Base.iterate(iter::MyImageIterator, state=1)
    if state > length(iter)
        return nothing
    end

    (iter.classes[state], iter.images[state]), state+1
end

function get_img_from_data(data)
    @assert all(size(data) .== (targetsize..., 3))

    data = convert(Array{N0f8}, clamp.(data .+ 0.5, 0, 1))
    data = permutedims(data, (3, 1, 2))

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
    datas = Array{Float16, 3}[]
    labels = OneHotVector{UInt32}[]
    count = 0
    @showprogress for (classname, imgname) in images()
        if count >= num_images
            break
        end
        count += 1

        # imgs = Array{N0f8, 4}()

        img = imgload(joinpath(datadir, classname, imgname))
        img = crop_and_resize_image(img)
        
        # data1 = permutedims(channelview(img), (2, 3, 1))
        # imggray = channelview(Gray.(img))
        # data2 = batch([imggray for _ in 1:3])
        label = Flux.onehot(classname, sort(cd(readdir, datadir)))

        for i in 1:4
            rot = recenter(RotMatrix(i*pi/2), center(img))
            data = warp(img, rot, axes(img))
            data1 = permutedims(channelview(data), (2, 3, 1))
            imggray = channelview(Gray.(data))
            data2 = batch([imggray for _ in 1:3])
            push!(datas, data1 .- 0.5)
            push!(datas, data2 .- 0.5)
            push!(labels, label)
            push!(labels, label)
        end

    end
    return batch(datas), batch(labels)
end

if !any("runs" .== readdir("."))
    mkdir("runs")
end

if !any("output" .== readdir("."))
    mkdir("output")
end

if !any("flower" .== readdir("."))
    @warn "flower dataset missing"
end

@info "Loading data..."
# @load "preprocessed/images.jld2" imgs
data, labels = gen_dataset()
# data = data .- 0.5f0 # normalize

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

randinit(shape...) = randn(Float32, shape...)
function Convolutional()
    return Chain(
        Conv((4, 4), 3 => 3*16, stride=2, pad=2, init=randinit),
        x->leakyrelu.(x, 0.2f0),
        Conv((2, 2), 3*16 => 3*32, stride=1, pad=2, init=randinit),
        x->leakyrelu.(x, 0.2f0),
        MaxPool((4, 4)),
        Dropout(0.2),
        Conv((2, 2), 3*32 => 3*64, stride=1, pad=1, init=randinit),
        x->leakyrelu.(x, 0.2f0),
        MaxPool((4, 4)),
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

ConvModel() = Chain(Convolutional(), EndLayers(reduce(*, size(dummy_output)[1:3])))
struct ModelState
    model
    loss::Float32
end


function do_training()
    @info "Starting training"
    opt = Flux.setup(Adam(hparams.lr), convmodel)
    tb_logger = TBLogger("log/", min_level=Logging.Info)
    function mycb(epoch)
        loss = 0
        correct = 0
        incorrect = 0
        img_batch_idx_loss = Tuple{Int, Int, Float32}[]
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

        with_logger(tb_logger) do
            @info "Gross Loss" grossloss=loss
            loss = loss / (correct+incorrect)
            @info "Average Loss" avgloss=loss
            # @info "Gross correct: $correct; gross incorrect: $incorrect"
            acc = 100 * correct / (correct + incorrect)
            @info "Correct/Incorrect" correct=correct incorrect=incorrect
            @info "Accuracy" acc=acc
            # @info "average loss: $loss; accuracy: $acc"
        end

        state = ModelState(Flux.state(convmodel |> cpu), loss)

        jldsave("runs/convmodel_$(now())_$epoch.jld2"; modelstate=state)
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
convmodel = ConvModel() |> gpu
# if ispath("runs/convmodel.jld2")
#     @info "Loading model"
#     jldopen("runs/convmodel.jld2", "r") do file
#         Flux.loadmodel!(convmodel, file["modelstate"].model)
#         # if file["modelstate"].loss < state.loss
#         #     state = file["modelstate"]
#         # end
#     end
# else
#     exit
# end
# convmodel = convmodel |> gpu
do_training()
