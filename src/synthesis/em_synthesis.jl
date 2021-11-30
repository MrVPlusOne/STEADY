"""
Synthesize the dynamics using an Expectation Maximization-style loop.  
"""
function em_synthesis(
    regression_alg::AbstractRegerssionAlgorithm,
    sketch::MotionModelSketch,
    obs_data_list::Vector{<:ObservationData},
    train_split::Int, # the number of training trajectories
    dyn_guess::GaussianGenerator;
    obs_model::Function,
    sampler::PosteriorSampler,
    n_fit_trajs::Int, # the number of trajectories to use for fitting the dynamics
    iteration_callback = (data::NamedTuple{(:iter, :trajectories, :dyn_est)}) -> nothing,
    logger=Base.current_logger(),
    max_iters::Int=100,
)
    timer = TimerOutput()
    
    @smart_assert train_split < length(obs_data_list)
    n_valid = length(obs_data_list) - train_split
    output_types = [v.type for v in sketch.output_vars]
    output_names = [v.name for v in sketch.output_vars]

    sampler_states = [new_state(sampler) for _ in obs_data_list]

    function sample_data(motion_model)
        local systems = map(obs_data_list) do obs_data
            MarkovSystem(obs_data.x0_dist, motion_model, obs_model)
        end
        
        sample_posterior_parallel(sampler, systems, obs_data_list, sampler_states)
    end
    
    dyn_history = GaussianGenerator[]
    logp_history = Vector{Float64}[]
    dyn_est = dyn_guess

    try @progress "fit_dynamics_iterative" for iter in 1:max_iters+1
        motion_model = mk_motion_model(sketch, dyn_est)
        (; trajectories, log_obs) = @timeit timer "sample_data" sample_data(motion_model)
        trajectories::Matrix{<:Vector}

        push!(dyn_history, dyn_est)
        push!(logp_history, log_obs)

        @timeit timer "iteration_callback" begin
            iteration_callback((;iter, trajectories, dyn_est))
        end

        (iter == max_iters+1) && break

        comps_σ = collect(dyn_est.σs)
        n_fit_trajs = min(n_fit_trajs, size(trajectories, 1))

        sol = @timeit timer "fit_dynamics_sindy_validated" begin 
            fit_best_dynamics(
                regression_alg, sketch, trajectories, obs_data_list, train_split, comps_σ;
                n_fit_trajs)
        end

        (; dynamics, model_info, optimizer_info, display_info) = sol
        dyn_est = dynamics

        log_obs_train = mean(log_obs[1:train_split])
        log_obs_valid = mean(log_obs[train_split+1:end])

        Base.with_logger(logger) do
            @info "performance" log_obs_train log_obs_valid
            @info "model" model_info... log_step_increment=0
            @info "optimizer" optimizer_info... log_step_increment=0
        end
        @info "em_synthesis" iter log_obs_train log_obs_valid
        for (k, value) in pairs(display_info)
            @info "$k" value 
        end
    end # end for
    catch exception
        @warn "Synthesis early stopped by exception."
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println(stdout)
        end
    end # end try block
    display(timer)
    (; dyn_est, dyn_history, logp_history, timer)
end

function show_dyn_history(dyn_history; io=stdout, first_rows=5, max_rows=15, table_width=100)
    let rows=[], step=max((length(dyn_history)-first_rows)÷max_rows, 1)
        for (i, dyn) in enumerate(dyn_history)
            (i <= first_rows || i % step == 0) && push!(rows, dyn)
        end
        println(io, "=====Dynamics History=====")
        show(io, MIME"text/plain"(), DataFrame(rows), truncate=table_width)
    end
end