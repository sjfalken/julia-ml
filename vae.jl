
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

latent_dim = 2
batch_size = 16
dev = gpu
	
images = MLDatasets.MNIST(:train).features
# Normalize to [-1, 1]
image_tensor = reshape(images, 28, 28, :)
# Partition into batches
data = DataLoader(image_tensor |> dev, batchsize = batch_size)
# Reshape(args...) = Reshape(args)
# (r::Reshape)(x) = reshape(x, r.shape)
# Flux.@functor Reshape

encoder_first = Chain(
    # Reshape(28, 28, 1, :),
    x -> reshape(x, 28, 28, 1, :),
    Conv((4, 4), 1 => 16, relu; stride = 2, pad = 4),
    Conv((4, 4), 16 => 16, relu; stride = 2, pad = 1),
    flatten,
    Dense(8*8*16 => 32, relu),
    # Dense(latent_dim, 256, relu)
)

decode = Chain(
    Dense(latent_dim => 32, relu),
    Dense(32 => 8*8*16, relu),
    x-> reshape(x, 8, 8, 16, batch_size),
    ConvTranspose((4, 4), 16 => 16; stride = 2, pad = 1),
    ConvTranspose((4, 4), 16 => 1; stride = 2, pad = 3),
    x-> reshape(x, 28, 28, batch_size),
) |> dev

encode = Parallel(
    (x, y) -> cat(x, y; dims=2),
    mean = Chain(
        encoder_first,
        Dense(32 => latent_dim),
        x -> reshape(x, latent_dim, 1, :)
    ),
    logvar = Chain(
        encoder_first,
        Dense(32 => latent_dim),
        x -> reshape(x, latent_dim, 1, :)
    ),
) |> dev

mutable struct VAE
    encoder::Any
    decoder::Any
    last_loss::Float32
end

loss_kl(mu, logsig2) = -0.5 * sum(-mu .^ 2 - exp.(logsig2) .+ 1 + logsig2) / batch_size
loss_reconst(x̂, x) = -sum(logitbinarycrossentropy.(x̂, x)) / batch_size
# loss_reconst(x̂, x) = mse(x̂, x)

function (m::VAE)(x)
    # x = flatten(input)

    h = m.encoder(x)
    mu = h[:, 1, :]
    logsig2 = h[:, 2, :]
    z = mu + exp.(logsig2 / 2) .* dev(randn(Float32, latent_dim))

    x̂ = m.decoder(z)
    # reg = 0.01 * sum(x->sum(x.^2), Flux.params(encode, decode))
    elbo = -loss_kl(mu, logsig2) + loss_reconst(x̂, x)
    m.last_loss = -elbo #+ reg

    return reshape(sigmoid(x̂), 28, 28, :)
end

Flux.@functor VAE

Flux.trainable(v::VAE) = (; v.encoder, v.decoder)
model = VAE(encode, decode, typemax(Float32)) |> dev

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
save_cb = throttle(() -> jldsave("models/vae_$(now()).jld2"; model_state=state(model |> cpu), optim_state = optim), 300)


sample_data = first(data)[:, :, 1:16]
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
    img = jim(cat([sample_data[:, :, 1:2] |> cpu, model(sample_data)[:, :, 1:2] |> cpu]; dims=3))
    ep_cb(img, steps)
    save_cb()
end
