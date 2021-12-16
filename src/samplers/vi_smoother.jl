# posterior smoother based on variational inference
import Flux
using Flux: Dense, Chain, LayerNorm
using Flux: @functor, softplus
using Random
using TimerOutputs


"""
Normal Sampler.
"""
struct NormalSampler{M1, M2}
    loc_net::M1
    scale_net::M2
end
Flux.@functor NormalSampler

function (com::NormalSampler)(x)
    (; loc_net, scale_net) = com
    sample_normal((μ=loc_net(x), σ=softplus.(scale_net(x))))
end

@kwdef(
struct VIGuide{H0, X0S<:NormalSampler, DXS<:NormalSampler, RNN, XE, OE}
    rnn_h0::H0
    x0_sampler::X0S
    dx_sampler::DXS
    rnn::RNN
    x_encoder::XE
    obs_encoder::OE
end)
Flux.@functor VIGuide

function (guide::VIGuide)(observations::Vector, Δt::Float32, batch_size)
    (; rnn_h0, x0_sampler, dx_sampler, rnn, x_encoder, obs_encoder) = guide
    h = rnn_h0
    future_info = [
        begin
            h, o = rnn(h, obs_encoder(observations[t]))
            h
        end
        for t in length(observations):-1:1
    ] |> reverse

    x, logp = x0_sampler(repeat(future_info[1], 1, batch_size))
    trajectory = [
        let (dx, logp1) = dx_sampler(vcat(
                x_encoder(x), 
                repeat(future_info[t], 1, batch_size)))
            logp += logp1
            x += dx .* Δt
        end
        for t in 1:length(future_info)]
    (; trajectory, logp)
end

function mk_guide(x_dim, y_dim, h_dim)
    VIGuide(;
        rnn_h0 = zeros(Float32, h_dim), 
        x0_sampler = NormalSampler(
            Dense(h_dim, x_dim),
            Dense(h_dim, x_dim),
        ),
        dx_sampler = NormalSampler(
            Dense(2h_dim, x_dim),
            Dense(2h_dim, x_dim),
        ),
        rnn = Flux.GRUCell(h_dim, h_dim),
        x_encoder = Chain(
            LayerNorm(x_dim),
            Dense(x_dim, h_dim, tanh),
        ),
        obs_encoder = Chain(
            LayerNorm(y_dim),
            Dense(y_dim, h_dim, tanh),
        ),
    )
end

to_device(on_gpu) = on_gpu ? Flux.gpu : Flux.cpu

@kwdef(
struct VISmoother{STV, VTS, OTV, G<:VIGuide}
    guide::G
    on_gpu::Bool
    state_to_vec::STV=s -> convert(Vector{Float32}, s) |> to_device(on_gpu)
    vec_to_state::VTS=s -> convert(Vector{Float64}, Flux.cpu(s))
    obs_to_vec::OTV=s -> convert(Vector{Float32}, s) |> to_device(on_gpu)
end)

function (smoother::VISmoother)(observations, Δt::Float32, batch_size::Int)
    (; guide, state_to_vec, vec_to_state, obs_to_vec) = smoother
    obs_encs = encode_observations(smoother, observations)
    (; trajectory, logp) = guide(obs_encs, Δt, batch_size)
    trajectories = decode_trajectories(smoother, trajectory)
    (; trajectories, logp)
end

batch_repeat(n) = x -> repeat(x, [1 for _ in size(x)]..., n)

function encode_observations(smoother::VISmoother, observations)
    [repeat(smoother.obs_to_vec(o), 1, 1) for o in observations]
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
    n_samples_f::Function,
    anneal_schedule::Function = step -> 1.0,
    callback::Function= _ -> nothing,
)
    (; state_to_vec, vec_to_state, obs_to_vec, guide) = smoother
    guide_time = decode_time = log_joint_time = 0.0
    obs_encs = encode_observations(smoother, observations)
    for step in 1:n_steps
        batch_size = n_samples_f(step)::Int
        loss = () -> begin
            guide_time += @elapsed begin
                traj_encs, lp_guide = guide(obs_encs, Δt, batch_size)
            end
            log_joint_time += @elapsed begin
                lp_joint = log_joint(traj_encs)
            end
            w = anneal_schedule(step)
            @smart_assert 0 ≤ w ≤ 1
            (w * sum(lp_guide) - sum(lp_joint)) / batch_size
        end
        ps = Flux.params(guide)
        (; val, grad) = Flux.withgradient(loss, ps) # compute gradient
        elbo = -val
        if isfinite(elbo)
            Flux.update!(optimizer, ps, grad) # update parameters
        else
            @warn "elbo is not finite: $elbo, skip a gradient step."
        end
        callback((; step, elbo, batch_size))
    end
    time_stats = (; guide_time, decode_time, log_joint_time)
    time_stats
end


function logpdf_normal(μ, σ, x)
    T = eltype(μ)
    a = T(0.5log(2π))
    vs = @. -0.5f0 * ((x - μ) / σ)^2 - log(σ) - a
    sum(vs, dims=1)
end

let s=(5,2), μ = randn(s), σ = abs.(randn(s)), x = randn(s)
@smart_assert sum(logpdf_normal(μ, σ, x)) ≈ sum(
    logpdf(MvNormal(μ[:, i], σ[:, i]), x[:, i]) for i in 1:2)
end

function sample_normal((μ, σ)) 
    x = μ + σ .* randn!(zero(μ))
    logp = logpdf_normal(μ, σ, x)
    (; val=x, logp)
end