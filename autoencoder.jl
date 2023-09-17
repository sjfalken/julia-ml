

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
    Dense(28*28 => 256), 
    x -> tanh.(x),
    Dense(256 => 8), 
    x -> tanh.(x),
)

decoder = Chain(
    Dense(8 => 256), 
    x -> tanh.(x),
    Dense(256 => 28*28), 
    x -> tanh.(x),
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


steps = 0
last_steps_loss = 0
last_steps_img = 0

function logtbloss(epoch, loss, steps)
    with_logger(tblg) do
        @info "" epoch loss log_step_increment=(steps - last_steps_loss)
        global last_steps_loss = steps
        # @info "progress" img
    end
end

function logtbimg(img, steps)
    with_logger(tblg) do
        # @info "" epoch loss
        @info "progress" img log_step_increment=(steps - last_steps_img)
        global last_steps_img = steps
    end
end

it_cb = throttle((e, l, s) -> logtbloss(e, l, s), 1)
ep_cb = throttle((i, s) -> logtbimg(i, s), 5)

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



        it_cb(epoch, lossval, steps)
        
        # push!(losses, lossval)
        # dotblog(epoch, val)
        update!(optim, model, grads[1])

        global steps += 1

    end
    img = jim(cat([sample_data |> cpu, model(sample_data) |> cpu]; dims=3))
    ep_cb(img, steps)

    # plot(losses)
    # savefig("output/plot.png")
end

