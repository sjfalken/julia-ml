

# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
using MLDatasets
using Flux: Dense, Chain, Adam, gradient, throttle, withgradient
using Flux: flatten, mse, gpu, cpu, DataLoader
using Flux.Optimisers: setup
using Flux.Optimise: train!, update!
using CUDA
using Plots
using TensorBoardLogger, Logging
using MIRTjim: jim
Plots.set_default_backend!(:plotlyjs)
	
images = MLDatasets.MNIST(:train).features
# Normalize to [-1, 1]
image_tensor = reshape(@.(2f0 * images - 1f0), 28, 28, :)
# Partition into batches
data = DataLoader(image_tensor |> gpu, batchsize = 16)

encoder = Chain(
    flatten,
    Dense(28*28 => 4),   # activation function inside layer
)

decoder = Chain(
    Dense(4 => 28*28),   # activation function inside layer
    x -> reshape(x, 28, 28, :),
)

model = Chain(
    encoder,
    decoder
) |> gpu



f = open("log.txt", "w")
lg = Logging.SimpleLogger(f, Logging.Info)
tblg = TBLogger("tb_logs/run")
optim = setup(Adam(), model)  # will store optimiser momentum, etc.
losses = []

function logtbloss(epoch, loss)
    with_logger(tblg) do
        @info "" epoch loss
        # @info "progress" img
    end
end

function logtbimg(img)
    with_logger(tblg) do
        # @info "" epoch loss
        @info "progress" img
    end
end

it_cb = throttle((e, l) -> logtbloss(e, l), 1)
ep_cb = throttle((i) -> logtbimg(i), 5)

sample_data = first(data)[:, :, 1:2]

for epoch in 1:100
    with_logger(lg) do
        @info "epoch" epoch
        flush(f)
    end
    for d in data
        
        lossval, grads = withgradient(model) do m
            out = m(d)
            mse(out, d)
        end



        it_cb(epoch, lossval)
        
        # push!(losses, lossval)
        # dotblog(epoch, val)
        update!(optim, model, grads[1])

    end
    img = jim(cat([sample_data |> cpu, model(sample_data) |> cpu]; dims=3))
    ep_cb(img)

    # plot(losses)
    # savefig("output/plot.png")
end

