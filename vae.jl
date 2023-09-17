
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
using MLDatasets
using Flux: gradient, throttle, withgradient, relu, params, DataLoader
using Flux: flatten, mse, gpu, cpu, state, @functor
using Flux
using JLD2
using Flux.Optimisers: setup
using Flux.Optimise: train!, update!
using Flux.Losses: logitbinarycrossentropy
using CUDA
using Plots
using TensorBoardLogger, Logging
using MIRTjim: jim
using Printf
using Dates
# Plots.set_default_backend!(:plotlyjs)

latent_dim = 8
batch_size = 16
	
images = MLDatasets.MNIST(:train).features
# Normalize to [-1, 1]
image_tensor = reshape(images, 28, 28, :)
# Partition into batches
data = DataLoader(image_tensor |> gpu, batchsize = batch_size)


encoder_first = Chain(
    Dense(28*28 => 64, relu), 
)

decode = Chain(
    Dense(latent_dim => 64, relu),
    Dense(64 => 28*28),
) |> gpu

encode = Parallel(
    (x, y) -> cat(x, y; dims=2),
    mean = Chain(
        encoder_first,
        Dense(64 => latent_dim),
        x -> reshape(x, latent_dim, 1, :)
    ),
    logvar = Chain(
        encoder_first,
        Dense(64 => latent_dim),
        x -> reshape(x, latent_dim, 1, :)
    ),
) |> gpu

mutable struct VAE
    encoder::Any
    decoder::Any
    last_loss::Float32
end

loss_kl(mu, logsig2) = 0.5 * sum(-mu .^ 2 - exp.(logsig2) .+ 1 + logsig2; dims=1)
loss_reconst(x̂, x) = sum(logitbinarycrossentropy.(x̂, x); dims=1) / latent_dim
# loss_reconst(x̂, x) = mse(x̂, x)

function (m::VAE)(input)
    x = flatten(input)

    h = m.encoder(x)
    mu = h[:, 1, :]
    logsig2 = h[:, 2, :]
    z = mu + exp.(logsig2 / 2) .* gpu(randn(Float32, latent_dim))
    x̂ = m.decoder(z)
    m.last_loss = sum(loss_kl(mu, logsig2) + loss_reconst(x̂, x)) / batch_size

    return reshape(sigmoid(x̂), 28, 28, :)
end

Flux.@functor VAE

model = VAE(encode, decode, typemax(Float32))

Flux.trainable(v::VAE) = (; v.encoder, v.decoder)

f = open("log.txt", "w")
lg = Logging.SimpleLogger(f, Logging.Info)
tblg = TBLogger("tb_logs/run")
optim = setup(Adam(), model)  # will store optimiser momentum, etc.
# losses = []


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

sample_data = first(data)[:, :, 1:1]
# ps = params(encoder, decoder)

for epoch in 1:200
    with_logger(lg) do
        @info "epoch" epoch
        flush(f)
    end
    for x in data

        lossval, grads = withgradient(model) do m
            x̂ = m(x)
            return m.last_loss
        end



        it_cb(epoch, lossval, steps)
        
        # push!(losses, lossuuval)
        # dotblog(epoch, val)
        update!(optim, model, grads[1])

        global steps += 1

    end
    img = jim(cat([sample_data |> cpu, model(sample_data) |> cpu]; dims=3))
    ep_cb(img, steps)

    # plot(losses)
    # savefig("output/plot.png")
end

model_state = state(model |> cpu)
jldsave("models/vae_$(now()).jld2"; model_state)