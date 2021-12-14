# posterior smoother based on variational inference
import Flux
using Flux: Dense, Chain, LayerNorm
using Flux: @functor, softplus
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

function (com::Combiner)(x, h)
    (; x_to_h, combined_to_loc, combined_to_scale) = com

    h_combined = (h + x_to_h(x)) / 2
    (μ=combined_to_loc(h_combined), σ=combined_to_scale(h_combined))
end

function mk_combiner(x_dim, rnn_dim)
    Combiner(
        Chain(
            # LayerNorm(x_dim),
            Dense(x_dim, rnn_dim, tanh),
            # Dense(rnn_dim, rnn_dim, tanh),
        ),
        Flux.Dense(rnn_dim, x_dim),
        Flux.Dense(rnn_dim, x_dim, softplus)
    )
end


struct VIGuide{H0, M, RNN, Enc, CB<:Combiner}
    rnn_h0::H0
    x0_μ::M
    x0_σ_logit::M
    rnn::RNN
    obs_cencoder::Enc
    combiner::CB
end

function mk_guide(x_dim, y_dim, rnn_dim)
    combiner = mk_combiner(x_dim, rnn_dim)
    rnn_h0 = zeros(Float32, rnn_dim)
    x0_μ = zeros(Float32, x_dim)
    x0_σ_logit = zeros(Float32, x_dim)
    # obs_cencoder = Chain(
        # LayerNorm(y_dim),
        # Dense(y_dim, rnn_dim, tanh),
    # )
    obs_cencoder = identity
    rnn = Flux.GRUCell(y_dim, rnn_dim)
    VIGuide(rnn_h0, x0_μ, x0_σ_logit, rnn, obs_cencoder, combiner)
end

Flux.@functor VIGuide

function logpdf_normal(μ, σ, x)
    T = eltype(μ)
    a = T(log(2π))
    vs = @. -0.5f0 * ((x - μ) / σ)^2 - log(σ) - 0.5f0a
    sum(vs, dims=1)
end

let s=(5,2), μ = randn(s), σ = abs.(randn(s)), x = randn(s)
@smart_assert sum(logpdf_normal(μ, σ, x)) ≈ sum(
    logpdf(MvNormal(μ[:, i], σ[:, i]), x[:, i]) for i in 1:2)
end

function sample_normal((μ, σ)) 
    x = μ + σ .* randn!(zero(μ))
    logp = logpdf_normal(μ, σ, x)
    (; x, logp)
end


function (guide::VIGuide)(observations::Vector, Δt::Float32)
    (; rnn_h0, x0_μ, x0_σ_logit, rnn, obs_encoder, combiner) = guide
    h = rnn_h0
    outs = [
        begin
            h, o = rnn(h, obs_encoder(observations[t]))
            h
        end
        for t in length(observations):-1:1]

    outs = reverse(outs)
    batch_size = size(observations[1])[end]
    x, logp = sample_normal(
        (batch_repeat(batch_size)(x0_μ), batch_repeat(batch_size)(softplus.(x0_σ_logit))))
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
    vec_to_state::VTS=s -> convert(Vector{Float64}, Flux.cpu(s))
    obs_to_vec::OTV=s -> convert(Vector{Float32}, s) |> to_device(on_gpu)
end)

function (smoother::VISmoother)(observations, Δt::Float32, n_trajs::Int)
    (; guide, state_to_vec, vec_to_state, obs_to_vec) = smoother
    obs_encs = encode_observations(smoother, observations, n_trajs)
    (; trajectory, logp) = guide(obs_encs, Δt)
    trajectories = decode_trajectories(smoother, trajectory)
    (; trajectories, logp)
end

batch_repeat(n) = x -> fill(x, n) |> hcatreduce

function encode_observations(smoother::VISmoother, observations, n_trajs)
    [batch_repeat(n_trajs)(smoother.obs_to_vec(o)) for o in observations]
end

function decode_trajectories(smoother::VISmoother, trajectory_enc::Vector)
    n_trajs = size(trajectory_enc[1])[end]
    T = length(trajectory_enc)
    map(1:n_trajs) do i 
        map(1:T) do t 
            smoother.vec_to_state(trajectory_enc[t][:, i])
        end
    end
end

function train_guide!(
    smoother::VISmoother, log_joint, observations, Δt;
    optimizer,
    n_steps::Int,
    n_samples::Int=64,
    callback::Function= _ -> nothing,
)
    (; state_to_vec, vec_to_state, obs_to_vec, guide) = smoother
    batch_size = n_samples
    obs_encs = encode_observations(smoother, observations, batch_size)
    for step in 1:n_steps
        loss = () -> begin
            traj_encs, lp_guide = guide(obs_encs, Δt)
            trajs = decode_trajectories(smoother, traj_encs)
            lp_joint = sum(log_joint(tr, observations) for tr in trajs)
            (sum(lp_guide) - lp_joint) / batch_size
        end
        ps = Flux.params(guide)
        (; val, grad) = Flux.withgradient(loss, ps) # compute gradient
        Flux.update!(optimizer, ps, grad) # update parameters
        elbo = -val
        callback((; step, elbo))
    end
end