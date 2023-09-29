
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
using MLUtils
using Flux: gradient, throttle, withgradient, relu, params, DataLoader
using Flux: mse, gpu, cpu, state, @functor
using OneHotArrays
using Flux
using JLD2
using Flux.Optimisers: setup
using Flux.Optimise: train!, update!
using Flux.Losses: logitcrossentropy
using MFCC
using CUDA
using Plots
using TensorBoardLogger, Logging
using MIRTjim: jim
using Printf
using Dates
using ProgressMeter: @showprogress
# Plots.set_default_backend!(:plotlyjs)
PREPROCESSED = "preprocessed/audioMNIST"
TRAINFILE = "$PREPROCESSED/train.jld2"
SR = 16000

batch_size = 32

dev = gpu

@info "Loading data..."
mfcc_tensor = jldopen(TRAINFILE) do f
    f["samples"]
end
# audio_tensor = reshape(samples, sample_size, :)

data_tensor = reshape(mfcc_tensor, 98, 13, 1, 60*50*10)

sample_indices = 1:(60*50*10)
train_indices, test_indices = MLUtils.splitobs(sample_indices; at=0.9, shuffle=true)



# mfcc_size = size(mfcc()[1])..., 1
# data_tensor = reshape(mfcc_tensor, mfcc_size..., size(mfcc_tensor, 4))

z = ones(Int, 60*50)
raw_labels = vcat([i*z for i in 0:9]...)
onehot_labels =  onehotbatch(raw_labels, 0:9)

train_data, train_labels = data_tensor[:, :, :, train_indices], onehot_labels[:, train_indices]
test_data, test_labels = data_tensor[:, :, :, test_indices] |> dev, onehot_labels[:, test_indices] |> dev

# onehot_labels = hcat([onehot(i, 1:10) for i in raw_labels]...)

data = DataLoader((train_data, train_labels) |> dev, batchsize = batch_size)

# @info minimum(x->x, audio_tensor)

@info "Setting up model..."
audioconv = @autosize (98, 13, 1, batch_size) Chain(
    Conv((8, 4), 1 => 4, relu; stride = (4, 2), pad = 1),
    Conv((4, 2), 4 => 8, relu; stride = (4, 1), pad = 1),
    Conv((4, 2), 8 => 8, relu; stride = 1, pad = 1),
    MLUtils.flatten,
    Dense(_ => 128),
    Dense(128 => 10)
)

model = audioconv |> gpu

f = open("log.txt", "w")
lg = Logging.SimpleLogger(f, Logging.Info)
tblg = TBLogger("tb_logs/run")
optim = setup(Adam(), model)  # will store optimiser momentum, etc.
# losses = []


steps = 0
last_steps_loss = 0
last_steps_img = 0

function logtbloss(epoch)
    ŷ = model(test_data)
    y = test_labels

    display(y)
    display(ŷ)
    loss = logitcrossentropy(ŷ, y; dims=1)
    guess(i) = argmax((ŷ |> cpu)[:, i]) 
    actual(i) = onecold((y |> cpu)[:, i])
    accuracy = sum([guess(i) == actual(i) for i in axes(y, 2)]) / size(y, 2)

    with_logger(tblg) do
        @info "" epoch loss accuracy log_step_increment=(steps - last_steps_loss)
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

it_cb = throttle((e) -> logtbloss(e), 2)
ep_cb = throttle((i, s) -> logtbimg(i, s), 5)
save_cb = throttle(() -> jldsave("models/audio_$(now()).jld2"; model_state=state(model |> cpu), optim_state = optim), 300)

@info "Starting training..."
for epoch in 1:1000
    with_logger(lg) do
        @info "epoch" epoch
        flush(f)
    end
    @showprogress for (d, y) in data


        lossval, grads = withgradient(model) do m
            ŷ = m(d)

            return logitcrossentropy(ŷ, y; dims=1)
        end

        it_cb(epoch)
        
        update!(optim, model, grads[1])

        global steps += 1

    end
    # img = jim(cat([sample_data[:, :, 1:2] |> cpu, model(sample_data)[:, :, 1:2] |> cpu]; dims=3)) ep_cb(img, steps)
    # save_cb()
end
