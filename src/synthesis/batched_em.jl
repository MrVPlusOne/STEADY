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
    minibatch=32, # the number of examples in each learning step
    trajs_per_ex = 10,
    callback::Function=_ -> nothing,
    weight_decay=1.0f-4,
)
    gradient_time = optimization_time = callback_time = smoothing_time = 0.0
    n_examples = common_batch_size(x0_batch, obs_seq[1], control_seq[1])
    T = length(obs_seq)

    all_ps = Flux.params(motion_model.core)
    @smart_assert length(all_ps) > 0 "No parameters to optimize."
    @info "total number of array parameters: $(length(all_ps))"
    reg_ps = Flux.Params(collect(regular_params(motion_model.core)))
    @info "total number of regular parameters: $(length(reg_ps))"

    for step in 1:n_steps
        ex_ids = shuffle(1:n_examples)[1:minibatch]
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
        core_inputs = BatchTuple(core_in_set)
        core_outputs = BatchTuple(core_out_set)

        loss() = begin
            (; μs::BatchTuple, σs::BatchTuple) = motion_model.core(core_inputs)
            lps = map(logpdf_normal, μs.val, σs.val, core_outputs.val)
            -mean(sum(lps)::AbsMat)
        end

        step == 1 && loss() # just for testing
        gradient_time += @elapsed begin
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
        CUDA.synchronize(; blocking=true)
    end
    @info "Training finished ($n_steps steps)."
end