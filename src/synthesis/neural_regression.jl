using Flux: Flux, Dense, Chain, Dropout, ADAM, SkipConnection
using Flux.Data: DataLoader
using CUDA: CUDA

struct NeuralNetWrapper{Trans,NN}
    input_transform::Trans
    network::NN
end

function (gg::GaussianGenerator{names,D,<:NeuralNetWrapper})(in) where {names,D}
    (; μ_f, σs) = gg
    μ = in |> values |> to_svec |> μ_f.input_transform |> μ_f.network |> NamedTuple{names}
    map(values(μ), values(σs)) do m, s
        Normal(m, s)
    end |> NamedTuple{names} |> DistrIterator
end

sqnorm(x) = sum(abs2, x)

@kwdef struct NeuralRegression{Net,Opt} <: AbstractRegerssionAlgorithm
    network::Net
    optimizer::Opt
    regularizer::Function = params -> 1e-4 * sum(sqnorm, params)
    batchsize::Int = 64
    max_epochs::Int = 10_000
    patience::Int = 5
end

function fit_best_dynamics(
    alg::NeuralRegression,
    sketch::MotionModelSketch,
    (inputs, outputs)::Tuple{Vector{<:NamedTuple},Matrix{Float64}},
    (valid_inputs, valid_outputs)::Tuple{Vector{<:NamedTuple},Matrix{Float64}},
    comps_σ_guess::Vector{Float64},
)
    (; network, optimizer, regularizer, batchsize, max_epochs) = alg

    params = Flux.params(network)

    input_mat, input_transform =
        let (; transformed, μ, σ) = normalize_transform([collect(x) for x in inputs])
            μ_s = to_svec(μ)
            σ_s = to_svec(σ)
            hcatreduce(transformed), x -> (x .- μ_s) ./ σ_s
        end  # input_mat shape: (n_features, n_samples)
    target_mat = transpose(outputs)
    valid_input_mat = [input_transform(collect(x)) for x in valid_inputs] |> hcatreduce
    valid_target_mat = transpose(valid_outputs)
    @smart_assert size(target_mat, 2) == size(input_mat, 2)

    train_loader = DataLoader((input_mat, target_mat); shuffle=true, batchsize)
    valid_loader = DataLoader((valid_input_mat, valid_target_mat); shuffle=false, batchsize)

    train_epoch_losses = []
    valid_epoch_losses = []
    best_valid_loss = Inf
    best_epoch = 0
    epochs_trained = 0

    progress = Progress(
        max_epochs * length(train_loader);
        desc="neural regression",
        output=stdout,
        enabled=true,
        showspeed=true,
    )
    for epoch in 1:max_epochs
        epoch_loss = []
        for (x, y) in train_loader
            loss_f() =
                mean(abs2, ((network(x) .- y) ./ comps_σ_guess)) + regularizer(params)
            l, grad = Flux.withgradient(loss_f, params)
            push!(epoch_loss, l)
            Flux.Optimise.update!(optimizer, params, grad) # update parameters
            next!(progress; showvalues=[(:epoch, epoch)])
        end
        train_loss = mean(epoch_loss)
        valid_loss = mean(
            mean(abs2, ((network(x) .- y) ./ comps_σ_guess)) for (x, y) in valid_loader
        )

        push!(train_epoch_losses, train_loss)
        push!(valid_epoch_losses, valid_loss)
        epochs_trained += 1

        if valid_loss < best_valid_loss
            best_valid_loss = valid_loss
            best_epoch = epoch
        elseif epoch - best_epoch > alg.patience
            break
        end
    end

    output_names = [v.name for v in sketch.output_vars]
    σs = fit_model_σ(valid_loader, network)
    σs_nt = NamedTuple(zip(output_names, σs))
    dynamics = network_to_gaussian(network, σs_nt, input_transform)

    model_info = NamedTuple()
    optimizer_info = (;
        epochs_trained,
        train_loss=train_epoch_losses[end],
        valid_loss=valid_epoch_losses[end],
    )
    display_info = (; epochs_trained, train_epoch_losses, valid_epoch_losses)

    (; dynamics, model_info, optimizer_info, display_info)
end

function network_to_gaussian(network, σs::NamedTuple, input_transform)
    net = network |> compile_neural_nets
    μ_f = NeuralNetWrapper(input_transform, net)
    GaussianGenerator(μ_f, σs, (; network))
end

function fit_model_σ(data_loader, network)
    Δs = Matrix{Float64}[]
    for (x, y) in data_loader
        ŷ = network(x)
        push!(Δs, (ŷ - y))
    end
    Δ = hcatreduce(Δs)
    std(Δ; mean=zeros(size(Δ, 1)), dims=2)
end

# function sample_particles_batch!(
#     outputs, inputs, 
#     nm::GaussianMotionModel{<:GaussianGenerator{names, D, <:NeuralNetWrapper}}, 
#     u, Δt;
#     cache::Dict,
# ) where {names, D}
#     sketch = nm.sketch
#     (; μ_f, σs) = nm.core
#     network = μ_f.network

#     n_features = sketch.input_vars |> length
#     input_mat = get!(cache, :input_cache) do 
#         Matrix{Float64}(undef, n_features, length(inputs)) 
#     end
#     for i in 1:length(inputs)
#         input_mat[:, i] .= collect(sketch.inputs_transform(inputs[i], u))
#     end

#     output_mat = network(input_mat)
#     for i in eachindex(outputs)
#         x = inputs[i]
#         out = map(NTuple{D}(output_mat[:, i]), σs) do m, s
#             rand(Normal(m, s))
#         end |> NamedTuple{names}
#         outputs[i] = sketch.outputs_transform(x, out, Δt)
#     end
# end

"""
Try to optimize the test time performance of the model by converting 
matrices to static matrices.
"""
function compile_neural_nets(chain::Chain)
    Chain(map(compile_neural_nets, chain.layers)...)
end

function compile_neural_nets(layer::Dense)
    (; weight, bias, σ) = layer
    Dense(to_static_array(weight), to_static_array(bias), σ)
end

function compile_neural_nets(layer::LayerNorm)
    layer
end

function compile_neural_nets(sk::SkipConnection)
    SkipConnection(compile_neural_nets(sk.layers), sk.connection)
end

struct StaticDropout
    p::Float64
    function StaticDropout(p)
        @assert 0 ≤ p ≤ 1
        new(p)
    end
end

(d::StaticDropout)(xs) = begin
    map(xs) do x
        rand() < d.p ? zero(x) : x
    end
end

function compile_neural_nets(layer::Dropout)
    # StaticDropout(layer.p)
    identity
end