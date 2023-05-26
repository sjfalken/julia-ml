using Pkg

# Activate a new project environment in the current directory
Pkg.activate(".")
# Add the required packages to the environment
Pkg.add(["Images", "Flux", "MLDatasets", "CUDA", "Parameters"])
using Base.Iterators: partition
using Printf
using Statistics
using Random
using Images
using Flux: params, DataLoader
using Flux: Dense, Conv, flatten, Dropout, Chain, BatchNorm, relu, ConvTranspose, leakyrelu, cpu, gpu
using Flux.Optimise: update!
using Flux.Losses: logitbinarycrossentropy
using MLDatasets
using CUDA

Base.@kwdef struct HyperParams
    batch_size::Int = 128
    latent_dim::Int = 100
    epochs::Int = 25
    verbose_freq::Int = 1000
    output_dim::Int = 5
    disc_lr::Float64 = 0.0002
    gen_lr::Float64 = 0.0002
    device::Function = cpu
end

function load_MNIST_images(hparams)
    # images = MNIST.traintensor(Float32)
    images = MNIST(split=:train).features

    # Normalize the images to (-1, 1)
    normalized_images = @. 2f0 * images - 1f0
    image_tensor = reshape(normalized_images, 28, 28, 1, :)

    # Create a dataloader that iterates over mini-batches of the image tensor
    dataloader = DataLoader(image_tensor, batchsize=hparams.batch_size, shuffle=true)
    
    return dataloader
end

dcgan_init(shape...) = randn(Float32, shape) * 0.02f0

function Generator(latent_dim)
    Chain(
        Dense(latent_dim, 7*7*256, bias=false),
        BatchNorm(7*7*256, relu),

        x -> reshape(x, 7, 7, 256, :),

        ConvTranspose((5, 5), 256 => 128; stride = 1, pad = 2, init = dcgan_init, bias=false),
        BatchNorm(128, relu),

        ConvTranspose((4, 4), 128 => 64; stride = 2, pad = 1, init = dcgan_init, bias=false),
        BatchNorm(64, relu),

        # The tanh activation ensures that output is in range of (-1, 1)
        ConvTranspose((4, 4), 64 => 1, tanh; stride = 2, pad = 1, init = dcgan_init, bias=false),
    )
end
# Create a dummy generator of latent dim 100
generator = Generator(100)
noise = randn(Float32, 100, 3) # The last axis is the batch size

# Feed the random noise to the generator
gen_image = generator(noise)
@assert size(gen_image) == (28, 28, 1, 3)

function Discriminator()
    Chain(
        Conv((4, 4), 1 => 64; stride = 2, pad = 1, init = dcgan_init),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.3),

        Conv((4, 4), 64 => 128; stride = 2, pad = 1, init = dcgan_init),
        x->leakyrelu.(x, 0.2f0),
        Dropout(0.3),

        # The output is now of the shape (7, 7, 128, batch_size)
        flatten,
        Dense(7 * 7 * 128, 1) 
    )
end

# Dummy Discriminator
discriminator = Discriminator()
# We pass the generated image to the discriminator
logits = discriminator(gen_image)
@assert size(logits) == (1, 3)

function discriminator_loss(real_output, fake_output)
    real_loss = logitbinarycrossentropy(real_output, 1)
    fake_loss = logitbinarycrossentropy(fake_output, 0)
    return real_loss + fake_loss
end

generator_loss(fake_output) = logitbinarycrossentropy(fake_output, 1)

function create_output_image(gen, fixed_noise, hparams)
    fake_images = cpu(gen.(fixed_noise))
    image_array = reduce(vcat, reduce.(hcat, partition(fake_images, hparams.output_dim)))
    image_array = permutedims(dropdims(image_array; dims=(3, 4)), (2, 1))
    image_array = @. Gray(image_array + 1f0) / 2f0
    return image_array
end

function train_discriminator!(gen, disc, real_img, fake_img, opt, ps, hparams)

    disc_loss, grads = Flux.withgradient(ps) do
        discriminator_loss(disc(real_img), disc(fake_img))
    end

    # Update the discriminator parameters
    update!(opt, ps, grads)
    return disc_loss
end

function train_generator!(gen, disc, fake_img, opt, ps, hparams)

    gen_loss, grads = Flux.withgradient(ps) do
        generator_loss(disc(fake_img))
    end

    update!(opt, ps, grads)
    return gen_loss
end

function train(hparams)

    dev = hparams.device
    # Check if CUDA is actually present
    if hparams.device == gpu
        if !CUDA.has_cuda()
        dev = cpu
        @warn "No gpu found, falling back to CPU"
        end
    end

    # Load the normalized MNIST images
    dataloader = load_MNIST_images(hparams)

    # Initialize the models and pass them to correct device
    disc = Discriminator() |> dev
    gen =  Generator(hparams.latent_dim) |> dev

    # Collect the generator and discriminator parameters
    disc_ps = params(disc)
    gen_ps = params(gen)

    # Initialize the ADAM optimisers for both the sub-models
    # with respective learning rates
    disc_opt = ADAM(hparams.disc_lr)
    gen_opt = ADAM(hparams.gen_lr)

    # Create a batch of fixed noise for visualizing the training of generator over time
    fixed_noise = [randn(Float32, hparams.latent_dim, 1) |> dev for _=1:hparams.output_dim^2]

    # Training loop
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for real_img in dataloader

            # Transfer the data to the GPU
            real_img = real_img |> dev

            # Create a random noise
            noise = randn!(similar(real_img, (hparams.latent_dim, hparams.batch_size)))
            # Pass the noise to the generator to create a fake imagae
            fake_img = gen(noise)

            # Update discriminator and generator
            loss_disc = train_discriminator!(gen, disc, real_img, fake_img, disc_opt, disc_ps, hparams)
            loss_gen = train_generator!(gen, disc, fake_img, gen_opt, gen_ps, hparams)

            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(train_steps), Discriminator loss = $(loss_disc), Generator loss = $(loss_gen)")
                # Save generated fake image
                output_image = create_output_image(gen, fixed_noise, hparams)
                save(@sprintf("output/dcgan_steps_%06d.png", train_steps), output_image)
            end
            train_steps += 1
        end
    end

    output_image = create_output_image(gen, fixed_noise, hparams)
    save(@sprintf("output/dcgan_steps_%06d.png", train_steps), output_image)

    return nothing
end

# Define the hyper-parameters (here, we go with the default ones)
hparams = HyperParams()
train(hparams)

folder = "output"
# Get the image filenames from the folder
img_paths = readdir(folder, join=true)
# Load all the images as an array
images = load.(img_paths)
# Join all the images in the array to create a matrix of images
gif_mat = cat(images..., dims=3)
save("./output.gif", gif_mat)