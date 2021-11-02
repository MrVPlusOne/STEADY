using Interpolations: LinearInterpolation

function landmark_readings(
    (; pos, θ), landmarks; sensor_range, σ_range, σ_bearing,
    range_falloff=1.0, bearing_only=true,
)
    DistrIterator(map(landmarks) do l
        l::AbstractVector
        rel = l - pos
        dis = norm_R2(rel)
        angle = atan(rel[2], rel[1]) - θ
        bearing = CircularNormal(angle, σ_bearing)
        core = if bearing_only
            (; bearing)
        else
            range = Normal(dis, σ_range)
            (; range, bearing)
        end
        p_detect = sigmoid(4(sensor_range - dis) / range_falloff)
        OptionalDistr(p_detect, DistrIterator(core))
    end)
end

"""
## Examples
```jldoctest
julia> variable_tuple(
           :a => ℝ(PUnits.Length),
           :v => ℝ(PUnits.Speed),
       ) |> typeof
NamedTuple{(:a, :v), Tuple{Main.SEDL.Var, Main.SEDL.Var}}
```
"""
function variable_tuple(vars::Pair{Symbol, PType}...)::NamedTuple
    NamedTuple(v => Var(v, ty) for (v, ty) in vars)
end

"""
The state needs to contain the properties `pos` and `θ`.
"""
function plot_2d_scenario!(
    states::AbsVec, obs_data, name::String; landmarks,
    marker_len=0.1, state_color=1, landmark_color=2, state_alpha=1.0,
)
    (; obs_frames, observations) = obs_data
    arrow_style = arrow(:closed, 0.001, 1.0)
    let
        @unzip xs, ys = map(x -> x.pos, states)
        plot!(xs, ys, label="Position ($name)", aspect_ratio=1.0; 
            linecolor=state_color, linealpha=state_alpha)
    end
    let
        markers = states[obs_frames]
        @unzip xs, ys = map(x -> x.pos, markers)
        dirs = map(markers) do x
            rotate2d(x.θ, @SVector[marker_len, 0.])
        end
        @unzip us, vs = dirs
        quiver!(xs, ys, quiver=(us, vs), arrow=arrow_style, arrowsize=0.01, 
            label="Orientation ($name)"; linecolor=state_color, linealpha=state_alpha)
    end
    isempty(landmarks) || let
        sensor_xs, sensor_ys = Float64[], Float64[]
        for t in obs_frames
            for (l, l_obs) in zip(landmarks, observations[t].landmarks)
                (l_obs === missing) && continue
                push!(sensor_xs, states[t].pos[1], l[1], NaN)
                push!(sensor_ys, states[t].pos[2], l[2], NaN)
            end
        end
        plot!(sensor_xs, sensor_ys, label="Sensor",
            linecolor=landmark_color, linealpha=0.5, linestyle=:dot)
        @unzip xs, ys = landmarks
        scatter!(xs, ys, label="Landmarks", color=landmark_color)
    end
end

function plot_2d_trajectories!(
    trajectories::AbsVec{<:AbsVec}, name::String; linealpha=0.2, linecolor=4,
)
    xs, ys = Float64[], Float64[]
    for tr in trajectories
        @unzip tr_xs, tr_ys = map(x -> x.pos, tr)
        append!(xs, tr_xs)
        append!(ys, tr_ys)
        push!(xs, NaN)
        push!(ys, NaN)
    end
    # plt = scatter!(xs, ys, label="particles ($name)", markersize=1.0, markerstrokewidth=0)
    plot!(xs, ys; label="Position ($name)", linecolor, linealpha)
end

function run_scenario(
    scenario::Scenario, true_params::NamedTuple, x0::NamedTuple, controller::Function; 
    times, obs_frames, comp_env::ComponentEnv, comps_guess, params_guess,
    max_ast_size=6,
    optim_options = Optim.Options(
        f_abstol=1e-4,
        iterations=100,
        time_limit=10.0,
    ),
    max_iters=501,
)
    vdata = variable_data(scenario, x0)
    x0_dist = init_state_distribution(vdata)
    sketch = dynamics_sketch(scenario)
    sketch_core = dynamics_core(scenario)
    true_motion_model = to_p_motion_model(sketch_core, sketch)(true_params)
    obs_dist = observation_dist(scenario)

    true_system = MarkovSystem(x0_dist, true_motion_model, obs_dist)

    @info("Enumerating programs...")
    pruner = IOPruner(; inputs=sample_rand_inputs(sketch, 100), comp_env)
    shape_env = ℝenv()
    senum = @time synthesis_enumeration(vdata, sketch, shape_env, comp_env, max_ast_size; pruner)
    display(senum)

    @info("Generating simulation data...")
    ex_data = simulate_trajectory(times, x0, true_system, controller)
    obs_data = (; times, obs_frames, ex_data.observations, ex_data.controls)

    let traj_p = plot(legend=:outerbottom, aspect_ratio=1.0)
        plot_scenario!(scenario, ex_data.states, obs_data, "truth"; 
            state_color=1, landmark_color=3)
    end |> display

    @info("Sampling posterior using the correct dynamics...")
    particle_sampler = ParticleFilterSampler(
        n_particles=60_000,
        n_trajs=12,
        n_runs=6,
        n_threads=6,
    )
    @time ffbs_result = sample_posterior(particle_sampler, true_system, obs_data)
    ffbs_trajs = ffbs_result.trajectories
    let plt = plot()
        plot_trajectories!(scenario, ffbs_trajs, "true posterior", linealpha=0.2, linecolor=2)
        plot_scenario!(scenario, ex_data.states, obs_data, "truth"; 
            state_color=1, landmark_color=3)
        # Car2D.plot_states!(map_result.states, "MAP states"; landmarks=[], obs_data, 
        #     state_color=:red, state_alpha=0.6)
        display("image/png", plt)
    end

    @info("Testing parameters fitting with the correct dynamics structure...")
    params_dist = params_distribution(sketch)
    check_params_logp(ex_data.states[1], x0_dist, true_params, params_dist)
    check_params_logp(ex_data.states[1], x0_dist, params_guess, params_dist)
    fit_r = @time fit_dynamics_params(
        WrappedFunc(to_p_motion_model(sketch_core, sketch)),
        ffbs_trajs,
        x0_dist, params_dist,
        obs_data,
        rand(params_dist),
        optim_options = Optim.Options(f_abstol=1e-4),
    )
    display(fit_r)

    @info("Performing iterative dynamics synthesis...")
    fit_settings = DynamicsFittingSettings(; 
        optim_options, evals_per_program=2, n_threads=Threads.nthreads())
    
    function iter_callback((; iter, trajectories, dyn_est))
        if iter <= 10 || mod1(iter, 10) == 1
            p = plot(title="iteration $iter")
            plot_trajectories!(scenario, trajectories, "estimated", linealpha=0.2, linecolor=2)
            plot_scenario!(scenario, ex_data.states, obs_data, "truth"; 
                state_color=1, landmark_color=3)
            display("image/png", p)
        end
    end

    iter_result = @time fit_dynamics_iterative(senum, obs_data, comps_guess, params_guess;
        obs_model=true_system.obs_model,
        sampler=particle_sampler,
        program_logp=prog_size_prior(0.2), fit_settings,
        max_iters,
        iteration_callback = iter_callback,
    )
    
    (; ex_data, obs_data, iter_result)
end