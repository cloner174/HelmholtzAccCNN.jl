#using Pkg
#Pkg.add([ "DataFrames", "CSV", "ImageFiltering", "Functors","Flux", "BSON", "Plots", "PyPlot", "KrylovMethods", "LaTeXStrings", "Distributions", "MLDatasets", "Images", "DelimitedFiles", "CUDA"])
#println("All packages are ready.")

using Dates
using BSON: @load
using Flux
using Functors
# --- 1. Include All Project Source Files ---
# This ensures all functions and types are available.
println("Loading project source files...")
include("src/flux_components.jl")
include("src/unet/utils.jl")
include("src/unet/model.jl")
include("src/multigrid/helmholtz_methods.jl")
include("src/unet/data.jl")
include("src/unet/train.jl")
include("src/kappa_models.jl")

# --- 2. Define Parameters for Initial Training ---
# These parameters are based on Section 5 of the paper.
println("Setting up training parameters...")

# Experiment Name
test_name = "initial_base_model_$(Dates.format(now(), "HH_MM_SS"))"

# Grid and Data Parameters
n = 608  # Grid points in first dimension [cite: 392]
m = 304  # Grid points in second dimension [cite: 392]
train_size = 2000 #20000 # Number of training samples [cite: 392]
test_size = 100 #1000   # Number of validation samples [cite: 392]

# Training Hyperparameters
iterations = 12 #120      # Number of epochs [cite: 392]
batch_size = 1       # Batch size [cite: 392]
init_lr = 0.0001      # Initial learning rate [cite: 392]
smaller_lr = 30       # Decrease learning rate every 30 epochs

# Physics and Data Generation Parameters
f = 10.0                          # Frequency (Hz)
omega = r_type(2.0 * pi * f)      # Angular frequency
gamma = zeros(r_type, n - 1, m - 1) |> pu # Attenuation term, zero for this case
kappa = ones(r_type, n - 1, m - 1) |> pu # Placeholder kappa, will be randomized
kappa_type = 1                    # Use random models from CIFAR10 dataset
e_vcycle_input = false            # Input is residual `r`, not V-Cycle error estimate
kappa_input = true                # The network uses kappa as an input
gamma_input = false               # The network does not use gamma as an input

# --- 3. Create the Encoder-Solver Model ---
# This creates the architecture from Figure 1 in the paper.
println("Creating the Encoder-Solver model...")

# The paper's architecture uses an Encoder-Solver (`FeaturesUNet` or `arch=2`)
# with `FFKappa` as the encoder and `FFSDNUnet` as the solver.
model = create_model!(
    e_vcycle_input, kappa_input, gamma_input;
    type = FFSDNUnet,      # The Solver U-Net
    k_type = FFKappa,      # The Encoder U-Net
    resnet_type = SResidualBlock, # The type of residual block to use
    k_chs = 2,
    arch = 2               # `arch=2` selects the Encoder-Solver (FeaturesUNet)
) |> cgpu

println("Model created successfully. Type: ", typeof(model))

println("\nStarting the initial training process.")
println("This will take a very long time, as noted in the paper.")
println("Training on $(train_size) samples for $(iterations) epochs...")

train_residual_unet!(
    model, test_name, n, m, f, kappa, omega, gamma,
    train_size, test_size, batch_size, iterations, init_lr;
    e_vcycle_input = e_vcycle_input,
    kappa_type = kappa_type,
    kappa_input = kappa_input,
    gamma_input = gamma_input,
    smaller_lr = smaller_lr,
    model_type = FFSDNUnet,
    k_type = FFKappa
)

# --- 5. Completion Message ---
println("\n-----------------------------------------------------")
println("Initial training complete.")
println("The base model has been saved to the 'models/' directory as '$(test_name).bson'")
println("You are now ready for Step 3.")
println("-----------------------------------------------------")