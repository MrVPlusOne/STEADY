@kwdef mutable struct EarlyStopping
    patience::Int
    model_info::Any = nothing
    best_loss::Real = Inf
    iters_waited::Int = 0
end

function (early_stopping::EarlyStopping)(current_loss::Real, model_info, save_model_f)
    if current_loss < early_stopping.best_loss
        early_stopping.best_loss = current_loss
        early_stopping.model_info = model_info
        early_stopping.iters_waited = 0
        save_model_f()
    else
        early_stopping.iters_waited += 1
    end
    should_stop = early_stopping.iters_waited >= early_stopping.patience
    (; should_stop)
end

function train_dynamics_EM!(
    motion_model::BatchedMotionModel,
    logpdf_obs::Function,
    x0_batch,
    obs_seq,
    control_seq,
    (; times, obs_frames);
    optimizer,
    n_steps::Int,
    lr_schedule=nothing,
    n_particles=100_000,
    examples_per_step=1, # the number of examples in each learning step
    trajs_per_ex=10,  # the number of posterior trajectories to draw per example
    obs_weight_schedule=step -> 1.0,  # returns a multiplier for the observation logpdf.
    callback::Function=_ -> (; should_stop = false),
    weight_decay=1.0f-4,
)
    n_examples = common_batch_size(x0_batch, obs_seq[1], control_seq[1])
    @smart_assert n_examples >= examples_per_step > 0
    T = length(obs_seq)

    all_ps = Flux.params(motion_model.core)
    @smart_assert length(all_ps) > 0 "No parameters to optimize."
    @info "total number of array parameters: $(length(all_ps))"
    reg_ps = Flux.Params(collect(regular_params(motion_model.core)))
    @info "total number of regular parameters: $(length(reg_ps))"

    for step in 1:n_steps
        ex_ids = shuffle(1:n_examples)[1:examples_per_step]
        obs_weight = obs_weight_schedule(step)
        log_obs_set = []
        core_in_set, core_out_set = BatchTuple[], BatchTuple[]
        foreach(ex_ids) do ex_id
            x0 = repeat(x0_batch[ex_id], n_particles)
            controls = getindex.(control_seq, ex_id)
            observations = getindex.(obs_seq, ex_id)
            local pf_result = batched_particle_filter(
                x0,
                (; times, obs_frames, controls, observations);
                motion_model,
                logpdf_obs=(args...) -> logpdf_obs(args...) * obs_weight,
                record_io=true,
                showprogress=false,
            )
            push!(log_obs_set, pf_result.log_obs)
            (; core_input_seq, core_output_seq) = batched_trajectories(
                pf_result, trajs_per_ex; record_io=true
            )
            append!(core_in_set, core_input_seq)
            append!(core_out_set, core_output_seq)
        end
        log_obs = mean(log_obs_set)
        n_trans = sum(x -> x.batch_size, core_in_set)
        Δt = times[2] - times[1]
        core = motion_model.core

        loss() = -transition_logp(core, core_in_set, core_out_set, Δt) / n_trans

        step == 1 && loss() # just for testing
        (; val, grad) = Flux.withgradient(loss, all_ps)
        isfinite(val) || error("Loss is not finite: $val")
        if lr_schedule !== nothing
            optimizer.eta = lr_schedule(step)
        end
            
        Flux.update!(optimizer, all_ps, grad) # update parameters
        for p in reg_ps
            p .-= weight_decay .* p
        end
        callback_args = (;
            step, loss=val, log_obs, obs_weight, lr=optimizer.eta
        )
        callback(callback_args).should_stop && break
    end
    @info "Training finished ($n_steps steps)."
end


"""
Simulatneous SLAM + dynamics learning using expectation-maximization.

Note: may need to fix the location of one of the landmarks if the observations are 
translational invariant.

## Parameters
- `x0_dists`: Should be a vector of distributions over initial states. These 
distributions will be optimized in the EM loop and should support 
`rand`, `logpdf`, and `Flux.params`.
"""
function train_dynamics_EM_SLAM!(
    motion_model::BatchedMotionModel,
    landmarks::AbstractArray,
    landmarks_to_logpdf_obs::Function,
    x0_dists::Vector,
    obs_seq,
    control_seq,
    (; times, obs_frames);
    optimizer,
    n_steps::Int,
    lr_schedule=nothing,
    n_particles=100_000,
    trajs_per_ex=10,  # the number of posterior trajectories to draw per example
    obs_weight_schedule=step -> 1.0,  # returns a multiplier for the observation logpdf.
    callback::Function=_ -> (; should_stop = false),
    weight_decay=1.0f-4,
)
    n_examples = length(x0_dists)
    tconf = motion_model.tconf

    all_ps = Flux.params(motion_model.core, landmarks, x0_dists)
    @smart_assert length(all_ps) > 0 "No parameters to optimize."
    @info "total number of array parameters: $(length(all_ps))"
    reg_ps = Flux.Params(collect(regular_params(motion_model.core)))
    @info "total number of regular parameters: $(length(reg_ps))"

    for step in 1:n_steps
        ex_id = rand(1:n_examples)
        obs_weight = obs_weight_schedule(step)
        x0 = tconf(rand(x0_dists[ex_id], n_particles)::BatchTuple)
        controls = getindex.(control_seq, ex_id)
        observations = getindex.(obs_seq, ex_id)
        log_obs_original = landmarks_to_logpdf_obs(landmarks)
        logpdf_obs = (args...) -> log_obs_original(args...) .* obs_weight
        local pf_result = batched_particle_filter(
            x0,
            (; times, obs_frames, controls, observations);
            motion_model,
            logpdf_obs,
            record_io=true,
            showprogress=false,
        )
        (; trajectory, core_input_seq, core_output_seq) = batched_trajectories(
            pf_result, trajs_per_ex; record_io=true
        )
        log_obs = pf_result.log_obs
        n_trans = sum(x -> x.batch_size, core_input_seq)
        Δt = times[2] - times[1]
        core = motion_model.core

        loss() = begin 
            log_initial = sum(logpdf(x0_dists[ex_id], trajectory[1]))
            log_transition = sum(transition_logp(core, core_input_seq, core_output_seq, Δt))
            log_obs = sum(logpdf_obs.(trajectory, observations)) |> sum
            -(log_transition + log_initial + log_obs) / n_trans
        end

        step == 1 && loss() # just for testing
        (; val, grad) = CUDA.@allowscalar Flux.withgradient(loss, all_ps)
        isfinite(val) || error("Loss is not finite: $val")
        if lr_schedule !== nothing
            optimizer.eta = lr_schedule(step)
        end
            
        Flux.update!(optimizer, all_ps, grad) # update parameters
        for p in reg_ps
            p .-= weight_decay .* p
        end
        callback_args = (;
            step, loss=val, log_obs, obs_weight, lr=optimizer.eta
        )
        callback(callback_args).should_stop && break
    end
    @info "Training finished ($n_steps steps)."
end

function input_output_from_trajectory(
    sketch::BatchedMotionSketch,
    state_seq::TimeSeries{<:BatchTuple},
    control_seq::TimeSeries{<:BatchTuple},
    times;
    test_consistency=false,
)
    @smart_assert length(state_seq) == length(control_seq)

    core_in, core_out = BatchTuple[], BatchTuple[]

    for t in 1:(length(state_seq) - 1)
        core_input = sketch.state_to_input(state_seq[t], control_seq[t])
        Δt = times[t + 1] - times[t]
        core_output = sketch.output_from_state(state_seq[t], state_seq[t + 1], Δt)
        if test_consistency
            state_pred = sketch.output_to_state(state_seq[t], core_output, Δt)
            foreach(
                keys(state_pred.val), state_pred.val, state_seq[t + 1].val
            ) do comp, x̂, x1
                @smart_assert x̂ ≈ x1 "Failed for component $comp at time $t"
            end
        end
        push!(core_in, core_input)
        push!(core_out, core_output)
    end

    return (core_in, core_out)
end

function train_dynamics_supervised!(
    motion_core,
    core_input::BatchTuple,
    core_output::BatchTuple,
    Δt;
    optimizer,
    n_steps::Int,
    lr_schedule=nothing,
    callback::Function=_ -> (; should_stop = false),
    max_batch_size=1024,
    weight_decay=1.0f-4,
)
    gradient_time = optimization_time = callback_time = smoothing_time = 0.0

    all_ps = Flux.params(motion_core)
    @smart_assert length(all_ps) > 0 "No parameters to optimize."
    @info "total number of array parameters: $(length(all_ps))"
    reg_ps = Flux.Params(collect(regular_params(motion_core)))
    @info "total number of regular parameters: $(length(reg_ps))"

    n_data = core_input.batch_size
    mini_batch_size = min(max_batch_size, n_data)

    steps_trained = 0
    for step in 1:n_steps
        ids = sample(1:n_data, mini_batch_size; replace=false)
        loss() =
            -transition_logp(motion_core, core_input[ids], core_output[ids], Δt) /
            core_input.batch_size

        step == 1 && loss() # just for testing
        gradient_time += @elapsed CUDA.@sync begin
            (; val, grad) = Flux.withgradient(loss, all_ps)
            isfinite(val) || error("Loss is not finite: $val")
        end
        if lr_schedule !== nothing
            optimizer.eta = lr_schedule(step)
        end
        optimization_time += @elapsed CUDA.@sync begin
            Flux.update!(optimizer, all_ps, grad) # update parameters
            for p in reg_ps
                p .-= weight_decay .* p
            end
        end
        time_stats = (; gradient_time, optimization_time, smoothing_time, callback_time)
        callback_args = (; step, loss=val, lr=optimizer.eta, time_stats)
        callback_time += @elapsed CUDA.@sync begin
            to_stop = callback(callback_args).should_stop
        end
        steps_trained += 1
        to_stop && break
        GC.gc(false)  # To avoid running out of memory on GPU.
    end
    @info "Training finished ($steps_trained / $n_steps steps trained)."
end
