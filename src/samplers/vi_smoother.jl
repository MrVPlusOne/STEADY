# posterior smoother based on variational inference
using Flux: Flux
using Flux: Dense, Chain, LayerNorm, SkipConnection, BatchNorm
using Flux: @functor, softplus
using Random
using TimerOutputs
using Zygote: Zygote
using Setfield
##-----------------------------------------------------------

export TensorConfig, tensor_type, check_type
"""
A configuration specifying which type (cpu or gpu) and floating type should be used
for array/tensor operations. 

## Usages
- Use `tconf(x)` to convert an object `x` (e.g. a number or an array) to having the 
desired configuration `tconf`.
- Use `check_type(tconf, x)` to check if the object `x` has the desired configuration.
"""
struct TensorConfig{on_gpu,ftype} end
TensorConfig(on_gpu, ftype=Float32) = TensorConfig{on_gpu,ftype}()

Base.show(io::IO, ::TensorConfig{on_gpu,ftype}) where {on_gpu,ftype} =
    print(io, (; on_gpu, ftype))

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
export BatchTuple, common_batch_size, inflate_batch, split_components
"""
A wrapper over a batch of data with named components.

## Usages
Assuming `batch::BatchTuple`:
- Use `batch.val` to access the underlying NamedTuple, in which each 
component is a batch of data, with the last dimension having the batch size.
- Use `tconf(batch)` to change the batch to a different `TensorConfig`. 
- Use `batch[idx]` to access a subset of the batch, e.g., `tconf[1]` or `tconf[3:6]`, 
both returns a new `BatchTuple`.
- Use `inflate_batch(batch)` to expand the underlying value to full batch size (if the batch 
dimension is 1).
- Use `common_batch_size(batchs...)` to get the common batch size of a list of `BatchTuple`.
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
@use_short_show BatchTuple

"""
Create from a batch of values.
"""
function BatchTuple(tconf::TensorConfig, values::Vector{<:NamedTuple})
    BatchTuple(tconf, length(values), map(tconf, hcatreduce(values)))
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

function Base.repeat(batch::BatchTuple, n::Int; inflate=false)
    @smart_assert batch.batch_size == 1
    nb = @set batch.batch_size = n
    if inflate
        nb = inflate(nb)
    end
    nb
end

"""
Split a matrix into a named tuple of matrices according to the given sizes.

## Examples
```jldoctest
julia> split_components(ones(4, 5), (a=2, b=1, c=1))
(a = [1.0 1.0 … 1.0 1.0; 1.0 1.0 … 1.0 1.0], b = [1.0 1.0 … 1.0 1.0], c = [1.0 1.0 … 1.0 1.0])
```
"""
function split_components(x::AbsMat, comp_sizes::NamedNTuple{names,Int}) where {names}
    @smart_assert size(x, 1) == sum(comp_sizes)

    i = 1
    comps = map(names) do k
        x[i:((i += comp_sizes[k]) - 1), :]
    end
    NamedTuple{names}(comps)
end

"""
Foreach component whose batch dimension size is 1, pad it along the batch dimension 
to grow to the full batch size.
"""
function inflate_batch(batch::BatchTuple)
    bs = batch.batch_size
    @set batch.val = map(batch.val) do v
        if size(v)[end] == 1
            repeat(v, bs)
        else
            @smart_assert size(v)[end] == bs
            v
        end
    end
end

function check_components(
    batch::BatchTuple, comp_sizes::NamedNTuple{names,Int}
) where {names}
    Zygote.ignore() do
        @smart_assert keys(batch.val) == keys(comp_sizes)
        foreach(batch.val, comp_sizes) do v, dim
            @smart_assert size(v, 1) == dim
        end
    end
end

function check_approx(batch1::BatchTuple, batch2::BatchTuple; kargs...)
    @smart_assert keys(batch1.val) == keys(batch2.val)
    foreach(keys(batch1.val)) do k
        isapprox(batch1.val[k], batch2.val[k]; kargs...) || error(
            "Faild for componenet $k.\n\tLeft = $(batch1.val[k])\n\tRight = $(batch2.val[k])",
        )
    end
end

export plot_batched_series
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
    times,
    series::TimeSeries{<:BatchTuple};
    truth::Union{Nothing,TimeSeries{<:BatchTuple}}=nothing,
    plot_width=600,
    plot_args...,
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

    series = TensorConfig(false).(series)
    template = series[1]
    if truth !== nothing
        truth = TensorConfig(false).(truth)
        truth_temp = truth[1]
        @smart_assert truth_temp.batch_size == 1
        @smart_assert keys(truth_temp.val) == keys(template.val)
    end

    subplots = []
    for k in keys(template.val)
        xs, ys, linealpha = to_plot_data((b -> b.val[k]).(series))
        n_comps = size(template.val[k], 1)
        linecolor = hcatreduce(1:n_comps)
        label = ["$k[$i]" for i in 1:n_comps] |> hcatreduce
        push!(subplots, plot(xs, ys; linealpha, linecolor, label))
        if truth !== nothing
            txs, tys, _ = to_plot_data((b -> b.val[k]).(truth))
            plot!(
                txs, tys; label=hcatreduce(["$k[$i] (truth)" for i in 1:n_comps]), linecolor
            )
        end
    end
    plot(
        subplots...;
        layout=(length(subplots), 1),
        size=(plot_width, 0.6plot_width * length(subplots)),
        left_margin=1.5cm,
        plot_args...,
    )
end
##-----------------------------------------------------------
export FluxLayer
"""
    FluxLayer(forward, Val(layer_name), trainable, [layer_info=trainable])

An flexible wrapper to represent an arbitrary trainable computation. This can be 
used to implement new forward computation logic without declaring a new struct.

- `forward::Function` should have the signature `forward(trainable)(args...) -> result`. i.e., 
it should be a higher-order function that first takes in the trainable parameters and then 
the input arguments of the layer. See the examples below.
- `trainable` can be either a Tuple or NamedTuple. 
- `layer_info=trainable` can be anything and will only be used to display the layer.

## Examples
```julia
julia> layer = FluxLayer(
           Val(:MyBilinear), 
           (w=rand(2,3),), 
           info=(x_dim = 3, y_dim = 2)
       ) do trainable
           (x, y) -> y' * trainable.w * x
       end
FluxLayer{MyBilinear}((x_dim = 3, y_dim = 2))

julia> layer(ones(3), ones(2))
2.7499240871541684

julia> Flux.params(layer)
Params([[0.6650776972679167 0.004201998819206465 0.7122138939453165; 
0.2132508860808494 0.17734661433873555 0.977832996702144]])
```
"""
struct FluxLayer{name,PS<:Union{Tuple,NamedTuple},F<:Function,Info}
    forward::F
    layer_name::Val{name}
    trainable::PS
    info::Info
end
Flux.@functor FluxLayer

FluxLayer(forward::Function, layer_name, trainable; info=trainable) =
    FluxLayer(forward, layer_name, trainable, info)

function Base.show(io::IO, ::Type{<:FluxLayer{name}}) where {name}
    print(io, "FluxLayer{$name, ...}")
end

function Base.show(io::IO, l::FluxLayer{name}) where {name}
    print(io, "FluxLayer{$name}($(l.info))")
end

function (l::FluxLayer)(args...)
    l.forward(l.trainable)(args...)
end

function Flux.trainable(l::FluxLayer)
    Flux.trainable(l.trainable)
end

##-----------------------------------------------------------
# batched motion model
export BatchedMotionSketch, BatchedMotionModel, transition_logp, observation_logp

@kwdef struct BatchedMotionSketch{states,controls,inputs,outputs,F1,F2,F3}
    state_vars::NamedNTuple{states,Int}
    control_vars::NamedNTuple{controls,Int}
    input_vars::NamedNTuple{inputs,Int}
    output_vars::NamedNTuple{outputs,Int}
    "state_to_input(state, control) -> core_input"
    state_to_input::F1
    "output_to_state(state, core_output, Δt) -> new_state"
    output_to_state::F2
    "output_from_state(state, next_state, Δt) -> core_output"
    output_from_state::F3
end
@use_short_show BatchedMotionSketch

"""
    motion_model(state::BatchTuple, control::BatchTuple, Δt) -> distribution_of_next_state

A probabilistic motion model that can be used to sample new states in batch.
The motion model is constructed from a sketch and a core. Sampling of the new state is 
performed inside some local coordinate frame using the core, and the sketch is used to 
transform between this local and the global coordinate frame.

See also: [`transition_logp`](@ref), [`observation_logp`](@ref)
"""
@kwdef struct BatchedMotionModel{TConf<:TensorConfig,SK<:BatchedMotionSketch,Core}
    tconf::TConf
    sketch::SK
    core::Core
end
@use_short_show BatchedMotionModel

Flux.trainable(bmm::BatchedMotionModel) = Flux.trainable(bmm.core)


function (motion_model::BatchedMotionModel{TC})(
    x::BatchTuple{TC}, u::BatchTuple{TC}, Δt::Real; test_consistency::Bool=false
) where {TC}
    bs = common_batch_size(x.batch_size, u.batch_size)
    sketch, core = motion_model.sketch, motion_model.core

    inputs = sketch.state_to_input(x, u)::BatchTuple{TC}
    check_components(inputs, sketch.input_vars)

    (; μs, σs) = core(inputs)
    μs::BatchTuple{TC}
    σs::BatchTuple{TC}

    sample_next_state(rng) =
        let
            @unzip out, lps = map(μs.val, σs.val) do μ, σ
                sample_normal(rng, μ, σ)
            end
            out_batch = BatchTuple(x.tconf, bs, out)
            check_components(out_batch, sketch.output_vars)
            next_state = sketch.output_to_state(x, out_batch, x.tconf(Δt))::BatchTuple{TC}
            (; next_state, out_batch, lp=sum(lps))
        end

    compute_logp(next_state::BatchTuple{TC}) =
        let
            out_batch = sketch.output_from_state(x, next_state, x.tconf(Δt))::BatchTuple{TC}
            check_components(out_batch, sketch.output_vars)
            lp = map(μs.val, σs.val, out_batch.val) do μ, σ, o
                logpdf_normal(μ, σ, o)
            end |> sum
            (; lp, out_batch)
        end

    if test_consistency
        (x1, out_batch1, lp1) = sample_next_state(Random.GLOBAL_RNG)
        (lp2, out_batch2) = compute_logp(x1)
        check_approx(out_batch1, out_batch2)
        @smart_assert(
            isapprox(lp1, lp2, rtol=0.001),
            "relative diff: $(norm(lp1-lp2)/max(norm(lp1),norm(lp2)))"
        )
    end

    GenericSamplable(rng -> sample_next_state(rng)[1], x1 -> compute_logp(x1)[1])
end

function transition_logp(
    model::BatchedMotionModel, x_seq::TimeSeries, u_seq::TimeSeries, Δt::Real
)
    @smart_assert length(x_seq) == length(u_seq)
    T = length(x_seq)
    map(1:(T - 1)) do t
        dist = model(x_seq[t], u_seq[t], Δt)
        logpdf(dist, x_seq[t + 1])
    end |> sum
end

function observation_logp(obs_model, x_seq::TimeSeries, obs_seq)
    map(x_seq, obs_seq) do x, o
        logpdf(obs_model(x), o)
    end |> sum
end

##-----------------------------------------------------------
# VI Guide
export VIGuide, mk_guide, train_guide!
"""
    guide(obs_seq, control_seq, Δt) -> (; trajectory, logp)

Use `mk_guide` to create a new guide with the default architecture and `train_guide!`
to train the guide using variational inference.
"""
@kwdef struct VIGuide{SK<:BatchedMotionSketch,X0S,GC,RNN,H0,OE}
    sketch::SK
    "x0_sampler(future_batch_1) -> (x0_batch, logp)"
    x0_sampler::X0S
    "guide_core(core_in_batch, future_batch) -> core_out_batch"
    guide_core::GC
    rnn::RNN
    rnn_h0::H0
    "obs_encoder(y_batch, u_batch) -> obs_encoding"
    obs_encoder::OE
end
Flux.@functor VIGuide
@use_short_show VIGuide

function (guide::VIGuide)(
    obs_seq::TimeSeries{<:BatchTuple{TC}},
    control_seq::TimeSeries{<:BatchTuple{TC}},
    Δt::Real;
    rng=Random.GLOBAL_RNG,
    test_consistency::Bool=false,
) where {TC}
    batch_size = common_batch_size(obs_seq[1], control_seq[1])
    tconf = TC()

    (; rnn_h0, x0_sampler, rnn, obs_encoder) = guide

    h = (rnn_h0::Trainable).array
    future_info =
        [
            begin
                obs_mat = obs_encoder(obs_seq[t], control_seq[t])::AbsMat
                h, o = rnn(h, obs_mat)
                BatchTuple(tconf, batch_size, (; future_info=o))
            end for t in length(obs_seq):-1:1
        ] |> reverse

    future1 = inflate_batch(future_info[1])
    x0, logp = x0_sampler(future1)
    x::BatchTuple{TC} = x0
    trans = map(2:length(future_info)) do t
        core = inputs -> guide.guide_core(inputs::BatchTuple, future_info[t])
        mm_t = BatchedMotionModel(tconf, guide.sketch, core)
        x_dist = mm_t(x, control_seq[t - 1], Δt; test_consistency)
        x = rand(rng, x_dist)
        logp += logpdf(x_dist, x)
        x
    end
    trajectory = vcat([x0], trans)

    (; trajectory, logp)
end

function compute_normal_transforms(
    sketch::BatchedMotionSketch,
    sample_states::Vector{<:BatchTuple},
    sample_controls::Vector{<:BatchTuple},
    sample_observations::Vector{<:BatchTuple},
    Δt::Real,
)
    function from_batches(batches)
        template = batches[1]
        ks = keys(template.val)
        @unzip_named (shifts, :shift), (scales, :scale) = map(ks) do k
            data = map(batches) do b
                eachcol(b.val[k])
            end |> Iterators.flatten |> collect
            NormalTransform(data)
        end
        NormalTransform(NamedTuple{ks}(shifts), NamedTuple{ks}(scales))
    end

    state_trans = from_batches(sample_states)
    control_trans = from_batches(sample_controls)
    obs_trans = from_batches(sample_observations)
    core_in_trans = map(sample_states, sample_controls) do x, u
        sketch.state_to_input(x, u)
    end |> from_batches

    core_out_trans =
        map(sample_states[1:(end - 1)], sample_states[2:end]) do x, x1
            sketch.output_from_state(x, x1, Δt)
        end |> from_batches

    (; state_trans, control_trans, obs_trans, core_in_trans, core_out_trans)
end

function mk_guide(;
    sketch::BatchedMotionSketch,
    h_dim,
    y_dim,
    sample_states::Vector{<:BatchTuple},
    sample_controls::Vector{<:BatchTuple},
    sample_observations::Vector{<:BatchTuple},
    Δt::Real,
    min_σ=1.0f-3,
)
    mlp(n_in, n_out, out_activation=identity) = Chain(
        SkipConnection(Dense(n_in, h_dim, tanh), vcat),
        Dense(h_dim + n_in, n_out, out_activation),
    )
    # mlp(n_in, n_out, out_activation=identity) = (Dense(n_in, n_out, out_activation))

    (; state_trans, control_trans, obs_trans, core_in_trans, core_out_trans) = compute_normal_transforms(
        sketch, sample_states, sample_controls, sample_observations, Δt
    )

    core_in_dim = sum(sketch.input_vars)
    core_out_dim = sum(sketch.output_vars)
    x_dim = sum(sketch.state_vars)
    u_dim = sum(sketch.control_vars)

    rnn_dim = 2h_dim
    u_enc_dim = h_dim ÷ 2
    y_enc_dim = rnn_dim - u_enc_dim

    x0_sampler = FluxLayer(
        Val(:x0_sampler),
        (
            center=mlp(rnn_dim, x_dim),
            scale=mlp(rnn_dim, x_dim, x -> max.(softplus.(x), min_σ)),
        ),
    ) do (; center, scale)
        (future::BatchTuple) -> begin
            local input = future.val.future_info
            local x_enc, logp = sample_normal((center(input), scale(input)))
            local x_new = state_trans(split_components(x_enc, sketch.state_vars))
            BatchTuple(future.tconf, future.batch_size, x_new), logp
        end
    end

    guide_core = FluxLayer(
        Val(:guide_core),
        (
            center=mlp(core_in_dim + rnn_dim, core_out_dim),
            scale=mlp(core_in_dim + rnn_dim, core_out_dim, x -> max.(softplus.(x), min_σ)),
        ),
    ) do (; center, scale)
        (core_input::BatchTuple, future::BatchTuple) -> begin
            local batch_size = common_batch_size(core_input, future)
            local core_val = inv(core_in_trans)(core_input.val)
            local input = vcat_bc(core_val..., future.val...; batch_size)
            local μ_data, σ_data = center(input), scale(input)
            local μs = map(
                (x, y) -> x .+ y,
                split_components(μ_data, sketch.output_vars),
                core_out_trans.shift,
            )
            local σs = map(
                (x, y) -> x .+ y,
                split_components(σ_data, sketch.output_vars),
                core_out_trans.scale,
            )
            map((; μs, σs)) do nt
                BatchTuple(core_input.tconf, batch_size, nt)
            end
        end
    end

    obs_encoder = FluxLayer(
        Val(:obs_encoder),
        (u_encoder=mlp(u_dim, y_enc_dim, tanh), y_encoder=mlp(y_dim, u_enc_dim, tanh)),
    ) do (; u_encoder, y_encoder)
        (obs::BatchTuple, control::BatchTuple) -> begin
            local y_enc = y_encoder(vcat(inv(obs_trans)(obs.val)...))
            local u_enc = u_encoder(vcat(inv(control_trans)(control.val)...))
            vcat_bc(y_enc, u_enc; batch_size=common_batch_size(obs, control))
        end
    end

    VIGuide(;
        sketch,
        x0_sampler,
        guide_core,
        rnn=Flux.GRUCell(rnn_dim, rnn_dim),
        rnn_h0=Trainable(false, 0.01randn(Float32, rnn_dim)),
        obs_encoder,
    )
end

"""
`vcat` with broadcasting along the batch dimension.
"""
vcat_bc(xs::AbsMat...; batch_size) = begin
    xs = map(xs) do x
        if size(x, 2) == batch_size
            x
        else
            @smart_assert size(x, 2) == 1
            repeat(x, 1, batch_size)
        end
    end
    vcat(xs...)
end

function train_guide!(
    guide::VIGuide,
    log_joint,
    obs_seq,
    control_seq,
    Δt;
    optimizer,
    n_steps::Int,
    n_samples_f::Function,
    anneal_schedule::Function=step -> 1.0,
    lr_schedule=nothing,
    callback::Function=_ -> nothing,
    weight_decay=1.0f-4,
)
    guide_time = log_joint_time = 0.0
    batch_size = common_batch_size(obs_seq[1], control_seq[1])
    T = length(obs_seq)

    all_ps = Flux.params(guide)
    reg_ps = Flux.Params(collect(regular_params(guide)))

    for step in 1:n_steps
        # batch_size = n_samples_f(step)::Int  fixme
        w = anneal_schedule(step)
        @smart_assert 0 ≤ w ≤ 1

        loss(; test_consistency=false) = begin
            guide_time += @elapsed begin
                traj_seq, lp_guide = guide(obs_seq, control_seq, Δt; test_consistency)
            end
            log_joint_time += @elapsed begin
                lp_joint = log_joint(traj_seq)
            end
            @smart_assert size(lp_guide) == (1, batch_size)
            @smart_assert size(lp_joint) == (1, batch_size)
            (w * sum(lp_guide) - sum(lp_joint)) / (batch_size * T)
        end
        step == 1 && loss(; test_consistency=true)  # just for testing
        (; val, grad) = Flux.withgradient(loss, all_ps) # compute gradient
        elbo = -val
        if isfinite(elbo)
            if lr_schedule !== nothing
                optimizer.eta = lr_schedule(step)
            end
            Flux.update!(optimizer, all_ps, grad) # update parameters
            for p in reg_ps
                p .-= weight_decay * p
            end
        else
            @warn "elbo is not finite: $elbo, skip a gradient step."
        end
        callback((; step, elbo, batch_size, annealing=w, lr=optimizer.eta))
    end
    time_stats = (; guide_time, log_joint_time)
    time_stats
end

function logpdf_normal(μ, σ, x)
    T = eltype(μ)
    a = log(T(2π))
    vs = @. -(abs2((x - μ) / σ) + a) / 2 - log(σ)
    if vs isa AbsMat
        sum(vs; dims=1)::AbsMat
    else
        vs
    end
end

let s = (5, 2), μ = randn(s), σ = abs.(randn(s)), x = randn(s)
    @smart_assert sum(logpdf_normal(μ, σ, x)) ≈
        sum(logpdf(MvNormal(μ[:, i], σ[:, i]), x[:, i]) for i in 1:2)
end

"""
    sample_normal([rng], μ, σ) -> (; val, logp)
Sample a noramlly distributed value along with the corresponding log probability density.
"""
function sample_normal(rng::Random.AbstractRNG, μ, σ)
    σ = max.(σ, eps(eltype(σ)))
    x = μ + σ .* randn!(rng, zero(μ))
    logp = logpdf_normal(μ, σ, x)
    (; val=x, logp)
end

sample_normal((μ, σ)) = sample_normal(Random.GLOBAL_RNG, μ, σ)

"""
Return an iterator of trainable parameters that require regularization.
You can also wrap an array `x` inside a [`Trainale`](@ref), which would allow you 
to specify whether `x` needs regularization.
"""
regular_params(layer::Union{VIGuide,FluxLayer,Chain,SkipConnection}) =
    Iterators.flatten(map(regular_params, Flux.trainable(layer)))

regular_params(layer::Dense) = (layer.weight,)

regular_params(layer::Flux.GRUCell) = (layer.Wi, layer.Wh)

regular_params(::Union{BatchedMotionSketch,Function,BatchNorm}) = tuple()

regular_params(tp::Union{Tuple,NamedTuple}) = Iterators.flatten(map(regular_params, tp))


"""
A wrapper over a trainalbe array parameter that carries the information about whether it 
requires regularization.
"""
struct Trainable{regular,M<:AbstractArray}
    is_regular::Val{regular}
    array::M
end
Trainable(regular::Bool, a::AbstractArray) = Trainable(Val(regular), a)

Flux.@functor Trainable

Base.show(io::IO, x::Trainable{regular}) where {regular} = begin
    print(io, "Trainable($(regular), $(x.array))")
end

regular_params(layer::Trainable{true}) = (layer.array,)
regular_params(layer::Trainable{false}) = tuple()