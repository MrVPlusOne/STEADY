# posterior smoother based on variational inference
using Random
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
Flux.@functor BatchedMotionModel

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
            isapprox(sum(lp1), sum(lp2), rtol=1e-3, atol=1e-2),
            "abs diff: $(lp1-lp2)\nrelative diff: $(norm(lp1-lp2)/max(norm(lp1),norm(lp2)))"
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
export VIGuide, mk_guide, train_guide!, mk_nn_motion_model
"""
    guide(obs_seq, control_seq, Δt) -> (; trajectory, logp)

Use `mk_guide` to create a new guide with the default architecture and `train_guide!`
to train the guide using variational inference.
"""
@kwdef struct VIGuide{SK<:BatchedMotionSketch,X0S,GC,RNN,H0,OE}
    sketch::SK
    "x0_sampler(future_batch_1) -> (x0_batch, logp)"
    x0_sampler::X0S
    "guide_core(core_in_batch, x_batch, future_batch) -> core_out_batch"
    guide_core::GC
    rnn::RNN
    rnn_h0::H0
    "obs_encoder(y_batch, u_batch) -> obs_encoding"
    obs_encoder::OE
end
Flux.@functor VIGuide
@use_short_show VIGuide

function (guide::VIGuide)(
    x0::BatchTuple{TC},
    obs_seq::TimeSeries{<:BatchTuple{TC}},
    control_seq::TimeSeries{<:BatchTuple{TC}},
    Δt::Real;
    rng=Random.GLOBAL_RNG,
    test_consistency::Bool=false,
) where {TC}
    batch_size = common_batch_size(obs_seq[1], control_seq[1])
    tconf = TC()

    (; rnn_h0, x0_sampler, rnn, obs_encoder) = guide

    h = map((x::Trainable) -> x.array, rnn_h0)
    future_info =
        [
            begin
                obs_mat = obs_encoder(obs_seq[t], control_seq[t])::AbsMat
                h, o = rnn(h, obs_mat)
                BatchTuple(tconf, batch_size, (; future_info=o))
            end for t in length(obs_seq):-1:1
        ] |> reverse

    # future1 = inflate_batch(future_info[1])
    logp = 0
    x0 = inflate_batch(x0)
    x::BatchTuple{TC} = x0
    trans = map(2:length(future_info)) do t
        core = inputs -> guide.guide_core(inputs::BatchTuple, x, future_info[t])
        mm_t = BatchedMotionModel(tconf, guide.sketch, core)
        x_dist = mm_t(x, control_seq[t - 1], Δt; test_consistency)
        x = rand(rng, x_dist)
        logp = logp .+ logpdf(x_dist, x)
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
                    local tensor = b.val[k]
                    Flux.unstack(tensor, ndims(tensor))
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

mlp_with_skip(n_in, n_out, out_activation=identity; h_dim) =
    FluxLayer(
        Val(:mlp),
        (layer1=Dense(n_in, h_dim ÷ 2), layer2=Dense(h_dim ÷ 2 * 3 + n_in, n_out)),
    ) do (; layer1, layer2)
        x -> begin
            local y1 = layer1(x)
            local a1 = vcat(x, sin.(y1), tanh.(y1), relu.(y1))
            out_activation.(layer2(a1))
        end
    end

function mk_guide(;
    sketch::BatchedMotionSketch,
    dynamics_core,
    h_dim,
    y_dim,
    normal_transforms,
    min_σ=1.0f-3,
)
    (; state_trans, control_trans, obs_trans, core_in_trans, core_out_trans) =
        normal_transforms

    mlp(args...) = mlp_with_skip(args...; h_dim)

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
            # dynamics_core=dynamics_core,
            center=mlp(core_in_dim + 2core_out_dim + x_dim + rnn_dim, core_out_dim),
            scale=mlp(
                core_in_dim + 2core_out_dim + x_dim + rnn_dim,
                core_out_dim,
                sigmoid, # x -> max.(softplus.(x), min_σ),
            ),
        ),
    ) do (; center, scale)
        (core_input::BatchTuple, state::BatchTuple, future::BatchTuple) -> begin
            local batch_size = common_batch_size(core_input, state, future)
            local dy_μs, dy_σs = dynamics_core(core_input)

            local μs_val = inv(core_out_trans)(dy_μs.val)
            local σs_val = map(./, dy_σs.val, core_out_trans.scale)
            local core_val = inv(core_in_trans)(core_input.val)
            local x_val = inv(state_trans)(state.val)

            local input = vcat_bc(
                μs_val...,
                σs_val...,
                core_val...,
                x_val...,
                future.val...;
                batch_size,
            )
            local μ_data, σ_data = center(input), scale(input)
            local μs = map(
                .+,
                dy_μs.val,
                map(.*, core_out_trans.scale, split_components(μ_data, sketch.output_vars)),
            )
            local σs = map(.*, dy_σs.val, split_components(σ_data, sketch.output_vars))
            map((; μs, σs)) do nt
                BatchTuple(core_input.tconf, batch_size, nt)
            end
        end
    end

    obs_encoder = FluxLayer(
        Val(:obs_encoder),
        (u_encoder=mlp(u_dim, y_enc_dim), y_encoder=mlp(y_dim, u_enc_dim)),
    ) do (; u_encoder, y_encoder)
        (obs::BatchTuple, control::BatchTuple) -> begin
            batch_size = common_batch_size(obs, control)
            local y_enc = y_encoder(vcat_bc(inv(obs_trans)(obs.val)...; batch_size))
            local u_enc = u_encoder(vcat_bc(inv(control_trans)(control.val)...; batch_size))
            vcat(y_enc, u_enc)
        end
    end

    VIGuide(;
        sketch,
        x0_sampler,
        guide_core,
        # rnn=Flux.GRUCell(rnn_dim, rnn_dim),
        # rnn_h0=Trainable(false, 0.01randn(Float32, rnn_dim)),
        rnn=Flux.LSTMCell(rnn_dim, rnn_dim),
        rnn_h0=(
            Trainable(false, 0.01randn(Float32, rnn_dim)),
            Trainable(false, 0.01randn(Float32, rnn_dim)),
        ),
        obs_encoder,
    )
end

function mk_nn_motion_model(;
    sketch::BatchedMotionSketch, tconf, h_dim, normal_transforms, min_σ=1.0f-3
)
    mlp(args...) = mlp_with_skip(args...; h_dim)

    core_in_dim = sum(sketch.input_vars)
    core_out_dim = sum(sketch.output_vars)

    (; core_in_trans, core_out_trans) = normal_transforms

    core =
        FluxLayer(
            Val(:nn_motion_model),
            (
                center=mlp(core_in_dim, core_out_dim),
                scale=mlp(core_in_dim, core_out_dim, x -> max.(softplus.(x), min_σ)),
            ),
        ) do (; center, scale)
            (core_input::BatchTuple) -> begin
                local (; batch_size) = core_input
                local core_val = inv(core_in_trans)(core_input.val)
                local input = vcat_bc(core_val...; batch_size)
                local μ_data, σ_data = center(input), scale(input)
                local μs = map(
                    (x, sh) -> x .+ sh,
                    split_components(μ_data, sketch.output_vars),
                    core_out_trans.shift,
                )
                local σs = map(
                    (x, s) -> x .* s,
                    split_components(σ_data, sketch.output_vars),
                    core_out_trans.scale,
                )
                map((; μs, σs)) do nt
                    BatchTuple(core_input.tconf, batch_size, nt)
                end
            end
        end |> (tconf.on_gpu ? Flux.gpu : Flux.cpu)
    BatchedMotionModel(tconf, sketch, core)
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
    motion_model,
    obs_model,
    x0_batch,
    obs_seq,
    control_seq,
    Δt;
    optimizer,
    n_steps::Int,
    anneal_schedule::Function=step -> 1.0,
    lr_schedule=nothing,
    callback::Function=_ -> nothing,
    weight_decay=1.0f-4,
)
    guide_time = dynamics_time = obs_time = callback_time = 0.0
    batch_size = common_batch_size(obs_seq[1], control_seq[1])
    T = length(obs_seq)

    all_ps = Flux.params((guide, motion_model))
    @info "total number of param tensors: $(length(all_ps))"
    reg_ps = Flux.Params(collect(regular_params(guide)))

    for step in 1:n_steps
        # batch_size = n_samples_f(step)::Int  fixme
        w = anneal_schedule(step)
        @smart_assert 0 ≤ w ≤ 1
        elbo = Ref{Any}(missing)

        loss(; test_consistency=false) = begin
            guide_time += @elapsed begin
                traj_seq, lp_guide = guide(x0_batch, obs_seq, control_seq, Δt; test_consistency)
            end
            dynamics_time += @elapsed begin
                lp_dynamics = transition_logp(motion_model, traj_seq, control_seq, Δt)
            end
            obs_time += @elapsed begin
                lp_obs = observation_logp(obs_model, traj_seq, obs_seq)
            end
            @smart_assert size(lp_guide) == (1, batch_size)
            @smart_assert size(lp_dynamics) == (1, batch_size)
            @smart_assert size(lp_obs) == (1, batch_size)
            obs_term = sum(lp_obs) / (batch_size * T)
            transition_term = (sum(lp_dynamics) - sum(lp_guide)) / (batch_size * T)
            elbo[] = obs_term + transition_term
            -(obs_term + w * transition_term)
        end
        step == 1 && loss(; test_consistency=true)  # just for testing
        (; val, grad) = Flux.withgradient(loss, all_ps) # compute gradient
        if isfinite(val)
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
        time_stats = (; guide_time, dynamics_time, obs_time, callback_time)
        cb_args = (;
            step,
            loss=val,
            elbo=elbo[],
            batch_size,
            annealing=w,
            lr=optimizer.eta,
            time_stats,
        )
        callback_time += @elapsed callback(cb_args)
    end
    time_stats = (; guide_time, dynamics_time, obs_time)
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
##-----------------------------------------------------------
regular_params(::BatchedMotionSketch) = tuple()

regular_params(layer::VIGuide) =
    Iterators.flatten(map(regular_params, Flux.trainable(layer)))