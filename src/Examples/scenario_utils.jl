using Interpolations: LinearInterpolation
using DrWatson
import REPL
using REPL.TerminalMenus

abstract type Scenario end

"""
Information related to 2D landmark observations.
"""
@kwdef(
struct LandmarkInfo{bo, N}
    landmarks::SVector{N,SVector{2, Float64}}
    σ_range::Float64=1.0
    σ_bearing::Float64=5°
    sensor_range::Float64=10.0
    "The width of the sigmoid function that gives the probability of having observations 
    near the sensor range limit."
    range_falloff::Float64=1.0
    "If true, will only include bearing (angle) but not range (distrance) readings."
    bearing_only::Val{bo}=Val(false)
end)

function landmark_readings((; pos, θ), linfo::LandmarkInfo{bo}) where bo
    (; landmarks, σ_bearing, σ_range, sensor_range, range_falloff) = linfo
    DistrIterator(map(landmarks) do l
        l::AbstractVector
        rel = l - pos
        dis = norm_R2(rel)
        angle = atan(rel[2], rel[1]) - θ
        bearing = CircularNormal(angle, σ_bearing)
        core = if bo
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
    arrow_style = arrow(:open, :head, 0.001, 1.0)
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

struct ScenarioSetup{
    A<:AbsVec{Float64}, B<:AbsVec{Int}, 
    X<:NamedTuple, Ctrl<:Function,
}
    times::A
    obs_frames::B
    x0::X
    controller::Ctrl
end

abstract type SynthesisAlgorithm end

struct SindySynthesis <: SynthesisAlgorithm 
    basis::Vector{<:CompiledFunc}
    sketch::SindySketch
    optimizer::STLSOptimizer
end

@kwdef(
struct EnumerativeSynthesis <: SynthesisAlgorithm 
    comp_env::ComponentEnv
    comps_guess::NamedTuple
    params_guess::NamedTuple
    max_ast_size::Int=6
    optim_options::Optim.Options = Optim.Options(
        f_abstol=1e-4,
        iterations=100,
        time_limit=10.0,
    )
end)

function simulate_scenario(
    scenario::Scenario,
    true_motion_model::Function,
    setups::Vector{<:ScenarioSetup};
    save_dir,
)
    if isdir(save_dir) 
        @warn("The save_dir '$save_dir' already exists.")
        # menu = RadioMenu(["yes", "no"], pagesize=3)
        # choice = request("Empty the save_dir and proceed?", menu)
        choice = "yes" # TODO: make this work for the REPL
        if choice == "yes"
            rm(save_dir, recursive=true)
        else
            error("Aborted.")
        end
    end
    mkdir(save_dir)

    truth_path = mkdir(joinpath(save_dir, "ground_truth"))
    obs_dist = observation_dist(scenario)

    @info("Generating simulation data...")
    @unzip true_systems, ex_data_list, obs_data_list, traj_plts = map(enumerate(setups)) do (i, setup)
        (; times, obs_frames, x0, controller) = setup
        x0_dist = initial_state_dist(scenario, x0)
        true_system = MarkovSystem(x0_dist, true_motion_model, obs_dist)
        ex_data = simulate_trajectory(times, x0, true_system, controller)
        obs_data = 
            (; times, obs_frames, ex_data.observations, ex_data.controls, x0_dist)
        traj_p = let 
            plot(legend=:outerbottom, aspect_ratio=1.0, title="Run $i")
            plot_scenario!(scenario, ex_data.states, obs_data, "truth";
                state_color=1, landmark_color=3)
        end
        fig_path=joinpath(truth_path, "run_$i.svg")
        savefig(traj_p, fig_path)
        display(traj_p)
        (true_system, ex_data, obs_data, traj_p)
    end
    (; true_systems, ex_data_list, obs_data_list, truth_path, setups)
end

function test_scenario(
    scenario::Scenario,
    (; true_systems, ex_data_list, obs_data_list, truth_path, setups),
    algorithm::SynthesisAlgorithm,
    particle_sampler::ParticleFilterSampler,
)
    @info("Sampling posterior using the correct dynamics...")
    @time sampler_result = sample_posterior_parallel(
        particle_sampler, true_systems, obs_data_list)
    post_trajs = sampler_result.trajectories
    for i in 1:length(true_systems)
        true_states = ex_data_list[i].states
        x0_dist = true_systems[i].x0_dist

        plt = plot(aspect_ratio=1.0, title="Run $i")
        plot_trajectories!(scenario, post_trajs[:, i], "true posterior", linealpha=0.1, linecolor=2)
        plot_scenario!(scenario, true_states, obs_data_list[i], "truth"; 
            state_color=1, landmark_color=3)
        display("image/png", plt)
        fig_path=joinpath(truth_path, "posterior_$i.svg")
        savefig(plt, fig_path)
    end

    if algorithm isa SindySynthesis
        @info("Synthesizing from true posterior using SINDy...")
        (; basis, sketch, optimizer) = algorithm
        output_types = [v.type for v in sketch.output_vars]
        inputs, outputs = @time construct_inputs_outputs(post_trajs, obs_data_list, sketch)
        (; comps, stats) = @time fit_dynamics_sindy(
            basis, inputs, outputs, output_types, optimizer)
        @assert length(sketch.output_vars) == length(comps)
        dyn_est = OrderedDict(zip(sketch.output_vars, comps))
        @info "test_scenario" dyn_est
        @info "test_scenario" stats
    elseif algorithm isa EnumerativeSynthesis
        (; params_guess) = algorithm
        @info("Testing parameters fitting with the correct dynamics structure...")
        sketch = dynamics_sketch(scenario)
        sketch_core = dynamics_core(scenario)
        # true_motion_model = to_p_motion_model(sketch_core, sketch)(true_params)
        params_dist = params_distribution(sketch)

        let
            x0_dist = obs_data_list[1].x0_dist
            x0 = setups[1].x0
            check_params_logp(x0, x0_dist, true_params, params_dist)
            check_params_logp(x0, x0_dist, params_guess, params_dist)
        end

        fit_r = @time fit_dynamics_params(
            WrappedFunc(to_p_motion_model(sketch_core, sketch)),
            post_trajs,
            params_dist,
            obs_data_list,
            rand(params_dist),
            optim_options = Optim.Options(f_abstol=1e-4),
        )
        display(fit_r)
    end
    nothing
end

function synthesize_scenario(
    scenario::Scenario,
    algorithm::EnumerativeSynthesis,
    (; true_systems, ex_data_list, obs_data_list, truth_path);
    n_fit_trajs=15,
    particle_sampler = ParticleFilterSampler(
        n_particles=60_000,
        n_trajs=100,
    ),
    max_iters=501,
)
    (; comp_env) = algorithm
    @info("Enumerating programs...")
    pruner = IOPruner(; inputs=sample_rand_inputs(sketch, 100), comp_env)
    shape_env = ℝenv()
    senum = @time synthesis_enumeration(vdata, sketch, shape_env, comp_env, max_ast_size; pruner)
    display(senum)

    @info("Performing iterative dynamics synthesis...")
    fit_settings = DynamicsFittingSettings(; 
        optim_options, evals_per_program=2, n_threads=Threads.nthreads())

    function iter_callback((; iter, trajectories, dyn_est))
        (iter <= 10 || mod1(iter, 10) == 1) || return

        iter_path = mkpath(joinpath(save_dir, "iterations/$iter"))

        for i in 1:length(true_systems)
            true_states = ex_data_list[i].states
            plt = plot(aspect_ratio=1.0, title="Run $i (iter=$iter)")
            plot_trajectories!(scenario, trajectories[:, i], "estimated", linealpha=0.1, linecolor=2)
            plot_scenario!(scenario, true_states, obs_data_list[i], "truth"; 
                state_color=1, landmark_color=3)
            fig_path=joinpath(iter_path, "posterior_$i.svg")
            savefig(plt, fig_path)
            i == 1 && display("image/png", plt) # only display the first one
        end
    end

    iter_result = @time fit_dynamics_iterative(
        senum, obs_data_list, comps_guess, params_guess;
        obs_model=obs_dist,
        sampler=particle_sampler,
        program_logp=prog_size_prior(0.5), 
        n_fit_trajs,
        fit_settings,
        max_iters,
        iteration_callback = iter_callback,
    )

    (; ex_data_list, obs_data_list, iter_result)
end

function transform_sketch_inputs(f::Function, sketch, ex_data, true_params)
    (; state_to_inputs, outputs_to_state_dist) = sketch
    (; states, controls) = ex_data
    map(zip(states, controls)) do (x, u)
        inputs = state_to_inputs(x, u)
        prog_inputs = merge(inputs, true_params)
        f(prog_inputs)
    end
end