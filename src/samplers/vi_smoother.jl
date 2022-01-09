# posterior smoother based on variational inference
using Random
using CUDA
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
    "output_to_state(state, core_output, Δt) -> next_state"
    output_to_state::F2
    "output_from_state(state, next_state, Δt) -> core_output"
    output_from_state::F3
end
@use_short_show BatchedMotionSketch

"""
    motion_model(state::BatchTuple, control::BatchTuple, Δt) -> 
        (; next_state, logp, core_input, core_output)

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
    x::BatchTuple{TC}, u::BatchTuple{TC}, Δt::Real; test_consistency=false
) where {TC}
    check_type(TC(), Δt)
    bs = common_batch_size(x.batch_size, u.batch_size)
    sketch, core = motion_model.sketch, motion_model.core

    core_input = sketch.state_to_input(x, u)::BatchTuple{TC}
    check_components(core_input, sketch.input_vars)

    (; μs::BatchTuple{TC}, σs::BatchTuple{TC}) = core(core_input)

    @unzip out, lps = map(sample_normal, μs.val, σs.val)
    out_batch = BatchTuple(x.tconf, bs, out)
    check_components(out_batch, sketch.output_vars)
    next_state = sketch.output_to_state(x, out_batch, Δt)::BatchTuple{TC}
    if test_consistency
        out_batch2 = sketch.output_from_state(x, next_state, Δt)::BatchTuple{TC}
        foreach(out_batch.val, out_batch2.val) do v1, v2
            @smart_assert v1 ≈ v2
        end
    end
    (; next_state, logp=sum(lps), core_input, core_output=out_batch)
end

function transition_logp(
    core,
    core_input_seq::TimeSeries{<:BatchTuple},
    core_output_seq::TimeSeries{<:BatchTuple},
)::Real
    @smart_assert length(core_input_seq) == length(core_output_seq)
    transition_logp(core, BatchTuple(core_input_seq), BatchTuple(core_output_seq))
end

function transition_logp(core, core_input::BatchTuple, core_output::BatchTuple)::Real
    (; μs::BatchTuple, σs::BatchTuple) = core(core_input)
    lps = map(logpdf_normal, μs.val, σs.val, core_output.val)
    sum(sum(lps)::AbsMat)
end

function observation_logp(
    obs_model, x_seq::TimeSeries{<:BatchTuple}, obs_seq::TimeSeries{<:BatchTuple}
)::Real
    @smart_assert length(x_seq) == length(obs_seq)
    sum(logpdf(obs_model(BatchTuple(x_seq)), BatchTuple(obs_seq))::AbsMat)
end
##-----------------------------------------------------------
# batched particle filtering
function batched_particle_filter(
    x0::BatchTuple,
    (; times, obs_frames, controls, observations);
    motion_model::BatchedMotionModel,
    obs_model,
    record_io=false,
    resample_threshold::Float64=0.5,
    showprogress=true,
)
    x0 = inflate_batch(x0)
    tconf = x0.tconf
    @smart_assert eltype(obs_frames) <: Integer
    T, N = length(times), x0.batch_size
    particles = Vector{typeof(x0)}(undef, T)
    particles[1] = x0
    if record_io
        core_inputs = []
        core_outputs = []
    end
    id_type = tconf.on_gpu ? CuArray : Array
    ancestors = fill(id_type(1:N), T)
    n_resampled = 0

    log_weights = new_array(tconf, N)
    log_weights .= -log(N)
    weights = exp.(log_weights)
    log_obs = 0.0
    bins_buffer = new_array(tconf, N)

    progress = Progress(
        T; desc="batched_particle_filter", output=stdout, enabled=showprogress
    )
    for t in 1:T
        if t in obs_frames
            lp = logpdf(obs_model(particles[t]), observations[t]::BatchTuple)
            @smart_assert length(lp) == N
            log_weights += reshape(lp, N)
            log_z_t = logsumexp(log_weights)
            log_weights .-= log_z_t
            weights .= exp.(log_weights)
            log_obs += log_z_t

            # optionally resample
            if effective_particles(weights) < N * resample_threshold
                indices = copy(ancestors[t])
                systematic_resample!(indices, weights, bins_buffer)
                # @smart_assert all(i -> 1 <= i <= N, collect(indices)) "Bad indices!"
                ancestors[t] = indices
                log_weights .= -log(N)
                particles[t] = particles[t][indices]
                if record_io && t > 1
                    core_inputs[t - 1] = core_inputs[t - 1][indices]
                    core_outputs[t - 1] = core_outputs[t - 1][indices]
                end
                n_resampled += 1
            end
        end

        if t < T
            Δt = times[t + 1] - times[t]
            (; next_state, core_input, core_output) = motion_model(
                particles[t], controls[t]::BatchTuple, Δt
            )
            particles[t + 1] = next_state
            if record_io
                push!(core_inputs, core_input)
                push!(core_outputs, core_output)
            end
        end
        next!(progress)
    end

    out = (; particles, weights, log_weights, ancestors, log_obs, n_resampled)
    if record_io
        merge(out, (; core_inputs, core_outputs))
    else
        out
    end
end

# This is used to replace the logsumexp that causes trouble on GPU
function simple_logsumexp(xs)
    mx = maximum(xs)
    log(sum(exp.(xs .- mx))) .+ mx
end

let rx = randn(100)
    @smart_assert simple_logsumexp(rx * 100) ≈ logsumexp(rx * 100)
end

"""
Sample full trajectories from a batched particle filter by tracing the ancestral lineages.
"""
function batched_trajectories(pf_result, n_trajs; record_io=false)
    (; particles::Vector{<:BatchTuple}, ancestors::Vector{<:AbsVec}, weights::AbsVec) =
        pf_result
    if record_io
        (; core_inputs, core_outputs) = pf_result
        core_input_seq = []
        core_output_seq = []
    end
    # first sample indices at the last time step according to the final weights
    indices = systematic_resample(weights, n_trajs)
    T = length(particles)
    trajectory = BatchTuple[particles[T][indices]]
    for t in (T - 1):-1:1
        indices = ancestors[t + 1][indices]
        push!(trajectory, particles[t][indices])
        if record_io && t > 1
            push!(core_input_seq, core_inputs[t - 1][indices])
            push!(core_output_seq, core_outputs[t - 1][indices])
        end
    end

    trajectory = reverse(trajectory)
    if record_io
        (; trajectory, core_input_seq, core_output_seq)
    else
        trajectory
    end
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
    @unzip trans, core_in_seq, core_out_seq = map(2:length(future_info)) do t
        core = inputs -> guide.guide_core(inputs::BatchTuple, x, future_info[t])
        mm_t = BatchedMotionModel(tconf, guide.sketch, core)
        (x1, lp, core_in, core_out) = mm_t(x, control_seq[t - 1], Δt; test_consistency)
        x = x1
        logp = logp .+ lp
        (x, core_in, core_out)
    end
    trajectory = vcat([x0], trans)

    (; trajectory, logp, core_in_seq, core_out_seq)
end

function compute_normal_transforms(
    sketch::BatchedMotionSketch,
    sample_states::Vector{<:BatchTuple},
    sample_controls::Vector{<:BatchTuple},
    sample_observations::Vector{<:BatchTuple},
    Δt::Real,
)
    function from_batches(batches)
        template = batches[1]::BatchTuple
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
        sketch.state_to_input(x, u)::BatchTuple
    end |> from_batches

    core_out_trans =
        map(sample_states[1:(end - 1)], sample_states[2:end]) do x, x1
            sketch.output_from_state(x, x1, Δt)
        end |> from_batches

    (; state_trans, control_trans, obs_trans, core_in_trans, core_out_trans)
end

mlp_with_skip(n_in, n_out, out_activation=identity; h_dim) =
# FluxLayer(
#     Val(:mlp), (layer1=Dense(n_in, h_dim ÷ 2), layer2=Dense(h_dim ÷ 2 + n_in, n_out))
# ) do (; layer1, layer2)
#     x -> begin
#         local y1 = layer1(x)
#         local a1 = vcat(x, relu.(y1))
#         out_activation.(layer2(a1))
#     end
# end
    Chain(Dense(n_in, h_dim, relu), Dense(h_dim, n_out, out_activation))

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

    # x0_sampler = FluxLayer(
    #     Val(:x0_sampler),
    #     (
    #         center=mlp(rnn_dim, x_dim),
    #         scale=mlp(rnn_dim, x_dim, x -> max.(softplus.(x), min_σ)),
    #     ),
    # ) do (; center, scale)
    #     (future::BatchTuple) -> begin
    #         local input = future.val.future_info
    #         local x_enc, logp = sample_normal((center(input), scale(input)))
    #         local x_new = state_trans(split_components(x_enc, sketch.state_vars))
    #         BatchTuple(future.tconf, future.batch_size, x_new), logp
    #     end
    # end
    x0_sampler = nothing

    guide_core = FluxLayer(
        Val(:guide_core),
        (
            gate=Dense(h_dim, core_out_dim, sigmoid),
            propose_left=mlp(core_in_dim, h_dim, tanh),
            propose_right=mlp(rnn_dim + x_dim, h_dim, tanh),
            mean1=Dense(core_in_dim, core_out_dim; init=zero_init),
            mean2=Dense(h_dim, core_out_dim; init=zero_init),
            scale=Dense(
                h_dim, core_out_dim, x -> max.(softplus.(x), min_σ); init=zero_init
            ),
        ),
    ) do (; gate, propose_left, propose_right, mean1, mean2, scale)
        (core_input::BatchTuple, state::BatchTuple, future::BatchTuple) -> begin
            local batch_size = common_batch_size(core_input, state, future)

            local left_input = vcat_bc(inv(core_in_trans)(core_input.val)...; batch_size)
            local right_input = vcat_bc(
                inv(state_trans)(state.val)..., future.val...; batch_size
            )
            local prop = propose_left(left_input) + propose_right(right_input)
            local g = gate(prop)
            local μ_data = (1 .- g) .* mean1(left_input) .+ g .* mean2(prop)
            local σ_data = scale(prop)

            local μs = core_out_trans(split_components(μ_data, sketch.output_vars))
            local σs = map(
                .*,
                core_out_trans.scale,
                split_components(σ_data, sketch.output_vars),
            )
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

zero_init(args...) = Flux.identity_init(args...; gain=0)

function mk_nn_motion_model(;
    sketch::BatchedMotionSketch, tconf, h_dim, normal_transforms, min_σ=1.0f-3
)
    mlp(args...) = mlp_with_skip(args...; h_dim)

    core_in_dim = sum(sketch.input_vars)
    core_out_dim = sum(sketch.output_vars)

    (; core_in_trans, core_out_trans) = normal_transforms

    core = FluxLayer(
        Val(:nn_motion_model),
        (
            mean1=Dense(core_in_dim, core_out_dim; init=zero_init),
            mean2=Chain(
                Dense(core_in_dim, h_dim, relu),
                Dense(h_dim, core_out_dim; init=zero_init),
            ),
            scale=Dense(
                core_in_dim,
                core_out_dim,
                x -> max.(softplus.(x), min_σ);
                init=zero_init,
            ),
        ),
    ) do (; mean1, mean2, scale)
        (core_input::BatchTuple) -> begin
            local (; batch_size) = core_input
            local core_val = inv(core_in_trans)(core_input.val)
            local input = vcat_bc(core_val...; batch_size)
            local μ_data = mean1(input) + mean2(input)
            local σ_data = scale(input)
            local μs = core_out_trans(split_components(μ_data, sketch.output_vars))
            local σs = map(
                .*,
                core_out_trans.scale,
                split_components(σ_data, sketch.output_vars),
            )
            map((; μs, σs)) do nt
                BatchTuple(core_input.tconf, batch_size, nt)
            end
        end
    end
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
    motion_model_core,
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
    guide_time = dynamics_time = gradient_time = obs_time = callback_time = 0.0
    batch_size = common_batch_size(x0_batch, obs_seq[1], control_seq[1])
    T = length(obs_seq)

    all_ps = Flux.params((guide, motion_model_core))
    @info "total number of array parameters: $(length(all_ps))"
    reg_ps = Flux.params((collect ∘ regular_params).((guide, motion_model_core)))
    @info "total number of regular parameters: $(length(reg_ps))"

    for step in 1:n_steps
        # batch_size = n_samples_f(step)::Int  fixme
        w = anneal_schedule(step)
        @smart_assert 0 ≤ w ≤ 1
        elbo = Ref{Any}(missing)

        loss(; test_consistency=false) = begin
            guide_time += @elapsed begin
                traj_seq, lp_guide, core_in_seq, core_out_seq = guide(
                    x0_batch, obs_seq, control_seq, Δt; test_consistency
                )
            end
            dynamics_time += @elapsed begin
                lp_dynamics =
                    transition_logp(motion_model_core, core_in_seq, core_out_seq)::Real
            end
            obs_time += @elapsed begin
                lp_obs = observation_logp(obs_model, traj_seq, obs_seq)::Real
            end
            @smart_assert size(lp_guide) == (1, batch_size)
            obs_term = lp_obs / (batch_size * T)
            transition_term = (lp_dynamics - sum(lp_guide)) / (batch_size * T)
            elbo[] = obs_term + transition_term
            -(obs_term + w * transition_term)
        end
        step == 1 && loss(; test_consistency=true)  # just for testing
        gradient_time += @elapsed begin
            (; val, grad) = Flux.withgradient(loss, all_ps) # compute gradient
        end

        isfinite(val) || error("Loss is not finite: $val")
        if lr_schedule !== nothing
            optimizer.eta = lr_schedule(step)
        end
        Flux.update!(optimizer, all_ps, grad) # update parameters
        for p in reg_ps
            p .-= weight_decay .* p
        end

        time_stats = (; guide_time, dynamics_time, obs_time, gradient_time, callback_time)
        callback_args = (;
            step,
            loss=val,
            elbo=elbo[],
            batch_size,
            annealing=w,
            lr=optimizer.eta,
            time_stats,
        )
        callback_time += @elapsed callback(callback_args)
    end
    @info "Training finished ($n_steps steps)."
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

# function logpdf_normal(μ::AbstractArray{Float32}, σ::AbstractArray{Float32}, x::AbstractArray{Float32})
#     # avoid numerical inaccruacies
#     logpdf_normal(Flux.f64(μ), Flux.f64(σ), Flux.f64(x)) |> Flux.f32
# end

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
    # if μ isa CuArray
    #     μ .+ σ .* CUDA.randn(size(μ))
    x = μ .+ σ .* randn!(zero(μ))
    logp = logpdf_normal(μ, σ, x)
    (; val=x, logp)
end

function sample_normal(μ, σ)
    sample_normal(Random.GLOBAL_RNG, μ, σ)
end

sample_normal((μ, σ)) = sample_normal(Random.GLOBAL_RNG, μ, σ)
##-----------------------------------------------------------
regular_params(::BatchedMotionSketch) = tuple()

regular_params(layer::VIGuide) =
    Iterators.flatten(map(regular_params, Flux.trainable(layer)))