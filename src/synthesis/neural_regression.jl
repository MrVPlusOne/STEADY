using Flux: Dense, ADAM
using Flux.Data: DataLoader
import Flux
import CUDA


@kwdef(
struct NeuralRegression{Net, Opt} <: AbstractRegerssionAlgorithm
    network::Net
    optimizer::Opt
    l2_λ::Float64=0.01 # L2 regularization
    batchsize::Int=64
    max_epochs::Int=10
end)

function fit_best_dynamics(
    alg::NeuralRegression,
    sketch::MotionModelSketch,
    (inputs, outputs)::Tuple{Vector{<:NamedTuple}, Matrix{Float64}},
    (valid_inputs, valid_outputs)::Tuple{Vector{<:NamedTuple}, Matrix{Float64}},
    comps_σ_guess::Vector{Float64},
)
    (; network, optimizer, l2_λ, batchsize, max_epochs) = alg
    σs = comps_σ_guess

    params = Flux.params(network)

    input_mat = [collect(x) for x in inputs] |> hcatreduce  # (n_features, n_samples)
    target_mat = transpose(outputs)
    valid_input_mat = [collect(x) for x in valid_inputs] |> hcatreduce
    valid_target_mat = transpose(valid_outputs)
    @smart_assert size(target_mat, 2) == size(input_mat, 2)

    train_loader = DataLoader((input_mat, target_mat), shuffle=true; batchsize)
    valid_loader = DataLoader((valid_input_mat, valid_target_mat), shuffle=false; batchsize)

    n_train, n_valid = length(inputs), length(valid_inputs)

    last_valid_loss = Inf
    epochs_trained = 0

    progress = Progress(max_epochs * length(train_loader), desc="fit_best_dynamics",
        output=stdout, enabled=true)
    Flux.trainmode!(network, true)
    for epoch in 1:max_epochs
        for (x, y) in train_loader
            loss_f = () -> mean(((network(x) .- y) ./ σs) .^ 2) + sum(sqnorm, params)
            gs = Flux.gradient(loss_f, params) # compute gradient
            Flux.Optimise.update!(optimizer, params, gs) # update parameters
            next!(progress)
        end

        Flux.trainmode!(network, false)
        (new_σs, new_loss) = fit_model_σ(valid_loader, network)
        improved = new_loss < last_valid_loss
        σs = new_σs
        last_valid_loss = new_loss
        epochs_trained += 1

        improved || break
    end

    output_names = [v.name for v in sketch.output_vars]
    σs_nt = NamedTuple(zip(output_names, σs))
    Flux.trainmode!(network, true) # to enable dropout during sampling
    dynamics = network_to_gaussian(network, σs_nt)
    σ_list = [Symbol(name, ".σ") => σ for (name, σ) in zip(output_names, σs)]

    model_info = NamedTuple()
    optimizer_info = (; epochs_trained, valid_loss=last_valid_loss)
    display_info = merge(model_info, optimizer_info)

    (; dynamics, model_info, optimizer_info, display_info)
end

sqnorm(x) = sum(abs2, x)

function network_to_gaussian(μ_f, σs::NamedTuple{names}) where names
    GaussianGenerator(x -> NamedTuple{names}(μ_f(to_svec(values(x)))), 
        σs, (network = μ_f,))
end

function fit_model_σ(data_loader, network)
    Δs = Matrix{Float64}[]
    for (x, y) in data_loader
        ŷ = network(x)
        push!(Δs, (ŷ - y))
    end
    Δ = hcatreduce(Δs)
    σs = std(Δ, mean=zeros(size(Δ, 1)), dims=2)
    loss = mean(abs2, (Δ ./ σs))

    return (; σs, loss)
end