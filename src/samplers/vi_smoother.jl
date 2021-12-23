# posterior smoother based on variational inference
using Flux: Flux
using Flux: Dense, Chain, LayerNorm, SkipConnection
using Flux: @functor, softplus
using Random
using TimerOutputs

##-----------------------------------------------------------
"""
A configuration specifying which type (cpu or gpu) and floating type should be used
for array/tensor operations. 

## Usages
- Use `tconf(x)` to convert an object `x` (e.g. a number or an array) to having the 
desired configuration `tconf`.
- Use `check_type(tconf, x)` to check if the object `x` has the desired configuration.
"""
struct TensorConfig{on_gpu,ftype}
    TensorConfig(on_gpu, ftype=Float32) = new{on_gpu,ftype}()
end

function Base.getproperty(::TensorConfig{on_gpu,ftype}, s::Symbol) where {on_gpu,ftype}
    if s === :on_gpu
        return on_gpu
    elseif s === :ftype
        return ftype
    else
        error("Unknown property: $s ")
    end
end

@inline function tensor_type(::TensorConfig{on_gpu,ftype}) where {on_gpu,ftype}
    (on_gpu ? CUDA.CuArray : Array){ftype}
end

(tconf::TensorConfig{on_gpu,ftype})(v::Real) where {on_gpu,ftype} = convert(ftype, v)
(tconf::TensorConfig{on_gpu,ftype})(a::AbstractArray{<:Real}) where {on_gpu,ftype} =
    convert(tensor_type(tconf), a)

function check_type(tconf::TensorConfig{on_gpu,ftype}, x::Real) where {on_gpu,ftype}
    @smart_assert x isa ftype
end

function check_type(tconf::TensorConfig{on_gpu,ftype}, x) where {on_gpu,ftype}
    desired_type = tensor_type(tconf)
    @smart_assert typeof(x) <: desired_type
end

##-----------------------------------------------------------
"""
A wrapper over a batch of data with named components.

## Usages
Assuming `batch::BatchTuple`:
- Use `batch.val` to access the underlying NamedTuple, in which each 
component is a batch of data, with the last dimension having the batch size.
- Use `tconf(batch)` to change the batch to a different `TensorConfig`. 
- Use `tconf[idx]` to access a subset of the batch, e.g., `tconf[1]` or `tconf[3:6]`, 
both returns a new `BatchTuple`.

"""
struct BatchTuple{C<:TensorConfig,V<:NamedTuple}
    tconf::C
    batch_size::Int
    val::V
    BatchTuple(
        tconf::TensorConfig{on_gpu}, batch_size::Int, val::NamedTuple
    ) where {on_gpu} = begin
        map(keys(val)) do k
            v = getproperty(val, k)
            check_type(tconf, v)
            v isa Real ||
                size(v)[end] == 1 ||
                size(v)[end] == batch_size ||
                error("Element $k has size $(size(v)), but batch size is $batch_size")
        end
        new{typeof(tconf),typeof(val)}(tconf, batch_size, val)
    end
end

function BatchTuple(tconf::TensorConfig, values::Vector{<:NamedTuple})
    BatchTuple(tconf, length(values), hcatreduce(values) |> to_device(tconf))
end

function Base.length(batch::BatchTuple)
    batch.batch_size
end

function Base.lastindex(batch::BatchTuple)
    batch.batch_size
end

function Base.getindex(batch::BatchTuple, ids)
    new_val = map(v -> batch_subset(v, ids), batch.val)
    BatchTuple(batch.tconf, length(ids), new_val)
end

function (tconf::TensorConfig)(batch::BatchTuple)
    if tconf == batch.tconf
        batch
    else
        new_val = map(tconf, batch.val)
        BatchTuple(tconf, batch.batch_size, new_val)
    end
end

function batch_subset(m::Union{Real,AbstractArray}, ids)
    r = if m isa Real
        m
    else
        s = size(m)
        if s[end] == 1
            m
        else
            colons = map(_ -> :, s)
            ids1 = ids isa Integer ? (ids:ids) : ids
            m[colons[1:(end - 1)]..., ids1]
        end
    end
    r::typeof(m)
end
##-----------------------------------------------------------
# Batch utils
common_batch_size(sizes::Integer...) = begin
    max_size = max(sizes...)
    all(sizes) do s
        s == max_size || s == 1 || error("Inconsistent batch sizes: $sizes")
    end
    max_size
end

common_batch_size(bts::BatchTuple...) = begin
    common_batch_size(map(x -> x.batch_size, bts)...)
end

"""
A general plotting function that plots a time series of `BatchedTuple`s. Each batch 
component will be plotted inside a separate subplot.

## Examples
```julia
let
    tconf = TensorConfig(false)
    times = 0:0.1:5
    series = [
        BatchTuple(tconf, 
            [
                (
                    pos=[randn(Float32), randn(Float32)+2], 
                    θ=[randn(Float32) * sin(Float32(t))],
                ) 
                for b in 1:3
            ]) 
        for t in times
    ]

    plot_batched_series(times, series)
end
```
"""
function plot_batched_series(
    times, series::TimeSeries{<:BatchTuple}; plot_width=600, plot_args...
)
    to_plot_data(ys::TimeSeries{<:Real}) = (; xs=times, ys, alpha=1.0)
    to_plot_data(ys::TimeSeries{<:AbsMat{<:Real}}) = begin
        local (dim, batch_size) = size(ys[1])
        local Num = eltype(ys[1])
        local T = length(ys)
        local x_out = repeat([times; NaN], batch_size)
        local y_out = Matrix{Num}(undef, batch_size * (T + 1), dim)
        for i in 1:batch_size
            for t in 1:T
                y_out[(i - 1) * (T + 1) + t, :] = ys[t][:, i]
            end
            y_out[i * (T + 1), :] .= NaN
        end
        (; xs=x_out, ys=y_out, alpha=1.0 / sqrt(batch_size))
    end

    template = series[1]
    @smart_assert !template.tconf.on_gpu

    subplots = []
    for k in keys(template.val)
        xs, ys, linealpha = to_plot_data((b -> b.val[k]).(series))
        push!(subplots, plot(xs, ys; title=k, linealpha))
    end
    plot(
        subplots...;
        layout=(length(subplots), 1),
        size=(plot_width, 0.6plot_width * length(subplots)),
        plot_args...,
    )
end
##-----------------------------------------------------------
"""
Normal Sampler.
"""
struct NormalSampler{M1,M2}
    loc_net::M1
    scale_net::M2
end

Flux.@functor NormalSampler
@use_short_show NormalSampler

function (com::NormalSampler)(x)
    (; loc_net, scale_net) = com
    sample_normal((μ=loc_net(x), σ=softplus.(scale_net(x))))
end

@kwdef struct BatchedMotionSketch{inputs,outputs,F1,F2}
    input_vars::NamedTuple{inputs,<:Tuple{Vararg{Int}}}
    output_vars::NamedTuple{outputs,<:Tuple{Vararg{Int}}}
    "to_NN_inputs(state, control) -> NN_inputs"
    state_to_input::F1
    "transform_NN_outputs(state, NN_outputs, Δt) -> new_state"
    output_to_state::F2
end
@use_short_show BatchedMotionSketch

"""
    motion_model(state::BatchTuple, control::BatchTuple, Δt) -> (; new_state::BatchTuple, logp)

A probabilistic motion model that can be used to sample new states in batch.
The motion model is constructed from a sketch and a core. Sampling of the new state is 
performed inside some local coordinate frame using the core, and the sketch is used to 
transform between this local and the global coordinate frame.
"""
@kwdef struct BatchedMotionModel{TConf<:TensorConfig,SK<:BatchedMotionSketch,Core}
    tconf::TConf
    sketch::SK
    core::Core
end
@use_short_show BatchedMotionModel



function (motion_model::BatchedMotionModel{TC})(
    x::BatchTuple{TC}, u::BatchTuple{TC}, Δt::Real
) where {TC}
    bs = common_batch_size(x.batch_size, u.batch_size)
    sketch, core = motion_model.sketch, motion_model.core

    inputs = sketch.state_to_input(x, u)::BatchTuple{TC}
    foreach(inputs.val, sketch.input_vars) do v, dim
        @smart_assert size(v, 1) == dim
    end

    (; μs, σs) = core(inputs)
    μs::BatchTuple{TC}
    σs::BatchTuple{TC}

    @unzip out, logps = map(μs.val, σs.val) do μ, σ
        sample_normal((μ, σ))
    end
    foreach(out, sketch.output_vars) do v, dim
        @smart_assert size(v, 1) == dim
    end

    out_batch = BatchTuple(x.tconf, bs, out)
    new_state = sketch.output_to_state(x, out_batch, x.tconf(Δt))::BatchTuple{TC}

    (; new_state, logp=sum(values(logps)))
end

struct ObsEncoder{M1<:AbsVec,M2,M3}
    missing_y_embedding::M1
    y_encoder::M2
    u_encoder::M3
end
Flux.@functor ObsEncoder


function (enc::ObsEncoder)(obs::Union{<:Some,Nothing}, control, batch_size)
    (; missing_y_embedding, u_encoder, y_encoder) = enc
    obs_encoding = if isnothing(obs)
        batch_repeat(batch_size)(missing_y_embedding)
    else
        y_encoder(obs.value)
    end
    control_encoding = batch_repeat(batch_size)(u_encoder(control))
    vcat(obs_encoding, control_encoding)
end

@kwdef(struct VIGuide{SK<:BatchedMotionSketch,H0,X0S,DXS,RNN,XE,OE<:ObsEncoder}
    sketch::SK
    rnn_h0::H0
    x0_sampler::X0S
    dx_sampler::DXS
    rnn::RNN
    x_encoder::XE
    obs_encoder::OE
end)
Flux.@functor VIGuide
@use_short_show VIGuide

function (guide::VIGuide)(observations::Vector, controls::Vector, Δt::Float32, batch_size)
    (; rnn_h0, x0_sampler, dx_sampler, rnn, x_encoder, obs_encoder) = guide
    sketch = guide.sketch
    h = rnn_h0
    future_info =
        [
            begin
                h, o = rnn(h, obs_encoder(observations[t], controls[t], 1))
                o
            end for t in length(observations):-1:1
        ] |> reverse

    max_x = 1e3
    x, logp = x0_sampler(repeat(future_info[1], 1, batch_size))
    trajectory = [
        begin
            inputs = sketch.to_NN_inputs(x, controls[t])
            dx_sampler
            (dx, logp1) = dx_sampler(
                vcat(x_encoder(x), repeat(future_info[t], 1, batch_size))
            )
            logp += logp1
            x = clamp.(x .+ dx .* Δt, -max_x, max_x)
        end for t in 1:length(future_info)
    ]
    (; trajectory, logp)
end

function mk_guide(; x_dim, y_dim, u_dim, h_dim)
    mlp(n_in, n_out, out_activation=identity) = Chain(
        SkipConnection(Dense(n_in, h_dim, tanh), vcat),
        Dense(h_dim + n_in, n_out, out_activation),
        # Dense(n_in, n_out, out_activation),
    )
    rnn_dim = 2h_dim
    u_enc_dim = h_dim ÷ 2
    y_enc_dim = rnn_dim - u_enc_dim
    VIGuide(;
        x0_sampler=NormalSampler(mlp(rnn_dim, x_dim), mlp(rnn_dim, x_dim)),
        dx_sampler=NormalSampler(mlp(h_dim + rnn_dim, x_dim), mlp(h_dim + rnn_dim, x_dim)),
        rnn=Flux.GRUCell(rnn_dim, rnn_dim),
        rnn_h0=0.01randn(Float32, rnn_dim),
        # rnn=Flux.LSTMCell(rnn_dim, rnn_dim),
        # rnn_h0=(0.01randn(Float32, rnn_dim), 0.01randn(Float32, rnn_dim)),
        x_encoder=Chain(LayerNorm(x_dim), mlp(x_dim, h_dim, tanh)),
        obs_encoder=ObsEncoder(
            0.01randn(Float32, y_enc_dim),
            mlp(u_dim, y_enc_dim, tanh),
            mlp(y_dim, u_enc_dim, tanh),
        ),
    )
end

to_device(on_gpu) = on_gpu ? Flux.gpu : Flux.cpu

@kwdef(
    struct VISmoother{STV,VTS,OTV,CTV,G<:VIGuide}
        guide::G
        on_gpu::Bool
        state_to_vec::STV = s -> convert(Vector{Float32}, s) |> to_device(on_gpu)
        vec_to_state::VTS = s -> convert(Vector{Float64}, Flux.cpu(s))
        obs_to_vec::OTV = map_optional(
            s -> convert(Vector{Float32}, s) |> to_device(on_gpu)
        )
        control_to_vec::CTV = s -> convert(Vector{Float32}, s) |> to_device(on_gpu)
    end
)

function (smoother::VISmoother)(observations, controls, Δt::Float32, batch_size::Int)
    (; guide) = smoother
    y_encs, u_encs = encode_observations(smoother, observations, controls)
    (; trajectory, logp) = guide(y_encs, u_encs, Δt, batch_size)
    trajectories = decode_trajectories(smoother, trajectory)
    (; trajectories, logp)
end

batch_repeat(n) = x -> repeat(x, [1 for _ in size(x)]..., n)

function encode_observations(smoother::VISmoother, observations, controls)
    (; obs_to_vec, control_to_vec) = smoother
    obs_to_vec.(observations), control_to_vec.(controls)
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
    smoother::VISmoother,
    log_joint,
    observations,
    controls,
    Δt;
    optimizer,
    n_steps::Int,
    n_samples_f::Function,
    anneal_schedule::Function=step -> 1.0,
    callback::Function=_ -> nothing,
)
    (; state_to_vec, vec_to_state, obs_to_vec, guide) = smoother
    guide_time = decode_time = log_joint_time = 0.0
    y_encs, u_encs = encode_observations(smoother, observations, controls)
    for step in 1:n_steps
        batch_size = n_samples_f(step)::Int
        w = anneal_schedule(step)
        @smart_assert 0 ≤ w ≤ 1

        loss =
            () -> begin
                guide_time += @elapsed begin
                    traj_encs, lp_guide = guide(y_encs, u_encs, Δt, batch_size)
                end
                log_joint_time += @elapsed begin
                    lp_joint = log_joint(traj_encs)
                end
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
        callback((; step, elbo, batch_size, annealing=w))
    end
    time_stats = (; guide_time, decode_time, log_joint_time)
    time_stats
end

function logpdf_normal(μ, σ, x)
    T = eltype(μ)
    a = log(T(2π))
    vs = @. -(abs2((x - μ) / σ) + a) / 2 - log(σ)
    if vs isa AbsMat
        sum(vs; dims=1)
    else
        vs
    end
end

let s = (5, 2), μ = randn(s), σ = abs.(randn(s)), x = randn(s)
    @smart_assert sum(logpdf_normal(μ, σ, x)) ≈
        sum(logpdf(MvNormal(μ[:, i], σ[:, i]), x[:, i]) for i in 1:2)
end

"""
    sample_normal((μ, σ)) -> (; val, logp)
Sample a noramlly distributed value along with the corresponding log probability density.
"""
function sample_normal((μ, σ))
    σ = max.(σ, eps(eltype(σ)))
    x = μ + σ .* randn!(zero(μ))
    logp = logpdf_normal(μ, σ, x)
    (; val=x, logp)
end