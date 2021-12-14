# posterior smoother based on variational inference
import Flux
using Flux: @functor, Dense, Chain, softplus
using Random


"""
The Combiner in https://pyro.ai/examples/dmm.html.
"""
struct Combiner{M1, M2, M3}
    x_to_h::M1
    combined_to_loc::M2
    combined_to_scale::M3
end

Flux.@functor Combiner

function (com::Combiner)(z, h)
    (; x_to_h, combined_to_loc, combined_to_scale) = com

    h_combined = (h + x_to_h(z)) / 2
    (μ=combined_to_loc(h_combined), σ=combined_to_scale(h_combined))
end

function mk_combiner(x_dim, rnn_dim)
    Combiner(
        Flux.Dense(x_dim, rnn_dim, tanh),
        Flux.Dense(rnn_dim, x_dim),
        Flux.Dense(rnn_dim, x_dim, softplus)
    )
end


struct VIGuide{H0, M, RNN, CB<:Combiner}
    rnn_h0::H0
    x0_μ::M
    x0_σ_logit::M
    rnn::RNN
    combiner::CB
end

function mk_guide(x_dim, y_dim, rnn_dim)
    combiner = mk_combiner(x_dim, rnn_dim)
    rnn_h0 = zeros(Float32, rnn_dim)
    x0_μ = zeros(Float32, x_dim)
    x0_σ_logit = zeros(Float32, x_dim)
    rnn = Flux.GRUCell(y_dim, rnn_dim)
    VIGuide(rnn_h0, x0_μ, x0_σ_logit, rnn, combiner)
end

Flux.@functor VIGuide

function logpdf_normal(μ, σ, x)
    T = eltype(μ)
    a = T(log(2π))
    vs = @. -0.5f0 * ((x - μ) / σ)^2 - log(σ) - 0.5f0a
    sum(vs, dims=1)
end

let μ = randn(5), σ = rand(5), x = randn(5)
@smart_assert logpdf_normal(μ, σ, x) ≈ logpdf(MvNormal(μ, σ), x)
end

function sample_normal((μ, σ)) 
    x = μ + σ .* randn!(zero(μ))
    logp = logpdf_normal(μ, σ, x)
    (; x, logp)
end


function (guide::VIGuide)(observations, Δt::Float32)
    (; rnn_h0, x0_μ, x0_σ_logit, rnn, combiner) = guide
    h = rnn_h0
    outs = [
        begin
            h, o = rnn(h, observations[t])
            o
        end
        for t in length(observations):-1:1]

    outs = reverse(outs)
    x, logp = sample_normal((x0_μ, softplus.(x0_σ_logit)))
    trajectory = [
        let (dx, logp1) = sample_normal(combiner(x, outs[t]))
            logp += logp1
            x += dx .* Δt
        end
        for t in 1:length(outs)]
    (; trajectory, logp)
end

to_device(on_gpu) = x -> on_gpu ? Flux.gpu(x) : Flux.cpu(x)

@kwdef(
struct VISmoother{STV, VTS, OTV, G<:VIGuide}
    guide::G
    on_gpu::Bool
    state_to_vec::STV=s -> convert(Vector{Float32}, s) |> to_device(on_gpu)
    vec_to_state::VTS=s -> convert(Vector{Float64}, s) |> Flux.cpu
    obs_to_vec::OTV=s -> convert(Vector{Float32}, s) |> to_device(on_gpu)
end)

function (smoother::VISmoother)(observations, Δt::Float32)
    (; guide, state_to_vec, vec_to_state, obs_to_vec) = smoother
    (; trajectory, logp) = guide(obs_to_vec.(observations), Δt)
    (; trajectory=vec_to_state.(trajectory), logp)
end

function train_guide!(
    smoother::VISmoother, log_joint, observations, Δt;
    optimizer,
    n_steps::Int,
    n_samples::Int=64,
    callback::Function= _ -> nothing,
)
    (; state_to_vec, vec_to_state, obs_to_vec, guide) = smoother
    obs_encs = map(obs_to_vec, observations)
    for step in 1:n_steps
        loss = () -> begin
            elbo = 0.0
            for i in 1:n_samples
                traj_encs, lp_guide = guide(obs_encs, Δt)
                traj = map(vec_to_state, traj_encs)
                lp_joint = log_joint(traj, observations)
                elbo += lp_joint - lp_guide[1]
            end
            -elbo / n_samples
        end
        ps = Flux.params(guide)
        gs = Flux.gradient(loss, ps) # compute gradient
        Flux.update!(optimizer, ps, gs) # update parameters
        elbo = -loss()
        callback((; step, elbo))
    end
end