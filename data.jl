
using Images: load as imgload, imresize, mosaicview, channelview
using ImageShow
using Base.Filesystem: cd, readdir
using Flux: onehot
using ProgressMeter: @showprogress
using JLD2: @save, @load
using ColorTypes: RGB
using FixedPointNumbers: N0f8
using Random: shuffle

datadir = "flower"
targetsize = (256, 256)

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

function run() 
    count = 0
    # data = Array{Float16, 4}(undef, (128, 128, 3, 0))
    # labels = Array{UInt8, 2}(undef, (5, 0))
    @showprogress for img in images()
        data = img[1]
        lbl = img[2]
        label = onehot(lbl, sort(cd(readdir, datadir)))
        @save "preprocessed/$count.jld2" data label
        
        count += 1
        # data = cat(data, onedata; dims=4)
        # labels = cat(labels, onehotlabel; dims=2)
    end
end

run()
