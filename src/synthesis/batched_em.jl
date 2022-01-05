function train_dynamics_em!(
    motion_model::BatchedMotionModel,
    obs_model,
    x0_batch,
    obs_seq,
    control_seq,
    (; times, obs_frames);
    optimizer,
    n_steps::Int,
    sampling_model=motion_model,
    lr_schedule=nothing,
    n_particles=100_000,
    trajs_per_step=1, # the number of examples in each learning step
    trajs_per_ex=10,
    callback::Function=_ -> nothing,
    weight_decay=1.0f-4,
)
    gradient_time = optimization_time = callback_time = smoothing_time = 0.0
    n_examples = common_batch_size(x0_batch, obs_seq[1], control_seq[1])
    @smart_assert n_examples >= trajs_per_step > 0
    T = length(obs_seq)

    all_ps = Flux.params(motion_model.core)
    @smart_assert length(all_ps) > 0 "No parameters to optimize."
    @info "total number of array parameters: $(length(all_ps))"
    reg_ps = Flux.Params(collect(regular_params(motion_model.core)))
    @info "total number of regular parameters: $(length(reg_ps))"

    for step in 1:n_steps
        ex_ids = shuffle(1:n_examples)[1:trajs_per_step]
        log_obs_set = []
        core_in_set, core_out_set = BatchTuple[], BatchTuple[]
        foreach(ex_ids) do ex_id
            x0 = repeat(x0_batch[ex_id], n_particles)
            controls = getindex.(control_seq, ex_id)
            observations = getindex.(obs_seq, ex_id)
            smoothing_time += @elapsed begin
                local pf_result = batched_particle_filter(
                    x0,
                    (; times, obs_frames, controls, observations);
                    motion_model=sampling_model,
                    obs_model,
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
        end
        log_obs = mean(log_obs_set)
        n_trans = sum(x -> x.batch_size, core_in_set)

        loss() = -transition_logp(motion_model.core, core_in_set, core_out_set) / n_trans

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
        callback_args = (; step, loss=val, log_obs, lr=optimizer.eta, time_stats)
        callback_time += @elapsed callback(callback_args)
    end
    @info "Training finished ($n_steps steps)."
end

function input_output_from_trajectory(
    sketch::BatchedMotionSketch,
    state_seq::TimeSeries{<:BatchTuple},
    control_seq::TimeSeries{<:BatchTuple},
    times,
) 
    @smart_assert length(state_seq) == length(control_seq)

    core_in, core_out = BatchTuple[], BatchTuple[]

    for t in 1:length(state_seq)-1
        core_input = sketch.state_to_input(state_seq[t], control_seq[t])
        state_rate = map(state_seq[t], state_seq[t+1]) do x, x1
            (x1 .- x) ./ (times[t+1] - times[t])
        end
        core_output = sketch.output_from_state_rate(state_seq[t], state_rate)
        push!(core_in, core_input)
        push!(core_out, core_output)
    end

    return (core_in, core_out)
end

function train_dynamics_supervised!(
    motion_core,
    core_input::BatchTuple,
    core_output::BatchTuple;
    optimizer,
    n_steps::Int,
    lr_schedule=nothing,
    callback::Function=_ -> nothing,
    weight_decay=1.0f-4,
)
    gradient_time = optimization_time = callback_time = smoothing_time = 0.0

    all_ps = Flux.params(motion_core)
    @smart_assert length(all_ps) > 0 "No parameters to optimize."
    @info "total number of array parameters: $(length(all_ps))"
    reg_ps = Flux.Params(collect(regular_params(motion_core)))
    @info "total number of regular parameters: $(length(reg_ps))"

    loss() = -transition_logp(motion_core, core_input, core_output) / core_input.batch_size

    for step in 1:n_steps
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
        callback_time += @elapsed callback(callback_args)
    end
    @info "Training finished ($n_steps steps)."
end