using Interpolations: LinearInterpolation
using DrWatson
import REPL
using REPL.TerminalMenus
import JSON
using TensorBoardLogger: TBLogger
using Serialization
using ProgressLogging: @withprogress, @logprogress

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
    p_detect_min::Float64=1e-4
    "If true, will only include bearing (angle) but not range (distrance) readings."
    bearing_only::Val{bo}=Val(false)
end)

function landmark_readings((; pos, θ), linfo::LandmarkInfo{bo}) where bo
    (; landmarks, σ_bearing, σ_range, sensor_range, range_falloff, p_detect_min) = linfo
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
        p_detect = clamp(p_detect, p_detect_min, 1-p_detect_min) # to avoid an infinite loop
        OptionalDistr(p_detect, DistrIterator(core))
    end)
end

"""
The L2 loss defined on the SE(2) manifold.
"""
L2_in_SE2(x1, x2) = norm(x1.pos - x2.pos)^2 + angular_distance(x1.θ, x2.θ)^2

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
    trajectories::AbsVec{<:AbsVec}, name::String; linecolor=4,
)
    linealpha=1.0 / sqrt(length(trajectories))
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
    data_list = map(enumerate(setups)) do (i, setup)
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
    @unzip true_systems, ex_data_list, obs_data_list, traj_plts = data_list
    (; true_systems, ex_data_list, obs_data_list, setups, save_dir)
end

function test_posterior_sampling(
    scenario::Scenario,
    motion_model::Function, 
    test_name::String,
    (; ex_data_list, obs_data_list, save_dir, setups),
    post_sampler::PosteriorSampler;
    state_L2_loss::Function, # (x, y) -> Float64
    generate_plots::Bool=true,
)    
    systems = map(setups) do setup
        x0_dist = initial_state_dist(scenario, setup.x0)
        obs_dist = observation_dist(scenario)
        MarkovSystem(x0_dist, motion_model, obs_dist)
    end
    result_dir = joinpath(save_dir, test_name) |> mkpath
    @info("[$test_name] Sampling posterior using the provided motion model...")
    @time sampler_result = sample_posterior_parallel(
        post_sampler, systems, obs_data_list)
    @info test_posterior_sampling sampler_result.n_effective
    post_trajs = sampler_result.trajectories
    
    @unzip RMSE_list, = map(1:length(systems)) do i
        true_states = ex_data_list[i].states
        x0_dist = systems[i].x0_dist
        run_trajs = post_trajs[:, i]

        if generate_plots
            plt = plot(aspect_ratio=1.0, title="Run $i")
            plot_trajectories!(scenario, run_trajs, "true posterior", linecolor=2)
            plot_scenario!(scenario, true_states, obs_data_list[i], "truth"; 
                state_color=1, landmark_color=3)
            display("image/png", plt)
            fig_path=joinpath(result_dir, "posterior_$i.svg")
            savefig(plt, fig_path)
        end

        # analyze numerical metrics
        RMSE = map(run_trajs) do tr
            @smart_assert length(tr) == length(true_states)
            map(tr, true_states) do s1, s2
                state_L2_loss(s1, s2)
            end |> mean
        end |> mean |> sqrt
        (; RMSE)
    end

    log_obs = sampler_result.log_obs |> mean
    metrics = OrderedDict(:RMSE=>mean(RMSE_list), :log_obs=>log_obs)
    open(joinpath(result_dir, "metrics.json"), "w") do io
        JSON.print(io, metrics, 4)
    end
    
    (; post_trajs, metrics)
end

function test_dynamics_fitting(
    scenario::Scenario,
    train_split::Int,
    post_trajs::Matrix,
    obs_data_list,
    algorithm::AbstractRegerssionAlgorithm,
    sketch,
    comps_σ::AbstractVector{Float64},
    n_fit_trajs,
)
    @smart_assert train_split < length(obs_data_list)

    if !(algorithm isa EnumerativeSynthesis)
        @info("Synthesizing from true posterior...")
        output_types = [v.type for v in sketch.output_vars]
        sol = fit_best_dynamics(
            algorithm, sketch, post_trajs, obs_data_list, train_split, comps_σ;
            n_fit_trajs)
        (; dynamics, display_info) = sol
        @smart_assert length(sketch.output_vars) == length(dynamics)
        @info "test_scenario" dynamics
        for (k, v) in pairs(display_info)
            println("$k: $(repr("text/plain", v))")
        end
    else
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
    train_split::Int,
    (; ex_data_list, obs_data_list, save_dir),
    reg_alg::AbstractRegerssionAlgorithm,
    sketch,
    dyn_guess;
    n_fit_trajs=15,
    post_sampler = ParticleFilterSampler(
        n_particles=60_000,
        n_trajs=100,
    ),
    max_iters=501,
)
    if isdir(joinpath(save_dir, "tb_logs"))
        @warn "Removing existing tensorboard logs..."
        rm(joinpath(save_dir, "tb_logs"), recursive=true)
    end
    logger = TBLogger(joinpath(save_dir, "tb_logs"))
    function iteration_callback((; iter, trajectories, dyn_est))
        (iter <= 10 || mod1(iter, 10) == 1) || return

        iter_path = mkpath(joinpath(save_dir, "iterations/$iter"))

        for i in 1:length(ex_data_list)
            true_states = ex_data_list[i].states
            plt = plot(aspect_ratio=1.0, title="Run $i (iter=$iter)")
            plot_trajectories!(scenario, trajectories[:, i], "estimated", linecolor=2)
            plot_scenario!(scenario, true_states, obs_data_list[i], "truth"; 
                state_color=1, landmark_color=3)
            fig_path=joinpath(iter_path, "posterior_$i.svg")
            savefig(plt, fig_path)
            plt_arg = [Symbol("run_$i") => plt]
            Base.with_logger(logger) do
                @info "posterior" plt_arg... log_step_increment=0
            end
            i == 1 && display("image/png", plt) # only display the first one
        end

        open(joinpath(iter_path, "dyn_est.txt"), "w") do io
            show(io, "text/plain", dyn_est)
        end
        open(joinpath(iter_path, "dyn_est.serial"), "w") do io
            serialize(io, dyn_est)
        end
    end
    obs_model = observation_dist(scenario)

    syn_result = 
        if !(reg_alg isa EnumerativeSynthesis)
            @info("Running EM synthesis...")
            @time em_synthesis(reg_alg, sketch, obs_data_list, train_split, dyn_guess; 
                obs_model, sampler=post_sampler, n_fit_trajs, 
                iteration_callback, logger, max_iters)
        else
            (; comp_env) = reg_alg
            @info("Enumerating programs...")
            pruner = IOPruner(; inputs=sample_rand_inputs(sketch, 100), comp_env)
            shape_env = ℝenv()
            senum = @time synthesis_enumeration(vdata, sketch, shape_env, comp_env, max_ast_size; pruner)
            display(senum)

            @info("Performing iterative dynamics synthesis...")
            fit_settings = DynamicsFittingSettings(; 
                optim_options, evals_per_program=2, n_threads=Threads.nthreads())

            @time fit_dynamics_iterative(
                senum, obs_data_list, dyn_guess, params_guess;
                obs_model,
                sampler=post_sampler,
                program_logp=prog_size_prior(0.5), 
                n_fit_trajs,
                fit_settings,
                max_iters,
                iteration_callback,
            )
        end

    syn_result
end

function analyze_motion_model_performance(
    scenario::Scenario,
    dyn_est::GaussianGenerator,
    true_mm::Function,
    baseline_mm::Function,
    test_setups::AbsVec{<:ScenarioSetup};
    save_dir,
    post_sampler,
    state_L2_loss,
    n_repeat::Int=5,
)
    test_save_dir=joinpath(save_dir, "test")
    trained_mm = let 
        sketch = sindy_sketch(scenario)
        mk_motion_model(sketch, dyn_est)
    end

    test_sim_result = simulate_scenario(
        scenario, true_mm, test_setups; save_dir=test_save_dir)

    motion_models = [
        ("baseline", baseline_mm), ("trained", trained_mm), ("true_model", true_mm)]
    
    prog_step = 0
    rows = @withprogress name="analyze_motion_model_performance" begin
        map(motion_models) do (name, mm)
            row = map(1:n_repeat) do i
                @logprogress prog_step/(length(motion_models) * n_repeat)
                prog_step+=1
                test_posterior_sampling(
                    scenario, mm, name, 
                    test_sim_result, post_sampler; 
                    state_L2_loss, generate_plots=i==1
                ).metrics
            end |> to_measurement
            (; name, row...)
        end
    end
    metric_table = DataFrame(rows)
    write(joinpath(test_save_dir, "metrics.txt"), repr(metric_table))
    @tagsave(joinpath(test_save_dir, "metrics.jld2"), @strdict(metric_table))

    metric_table
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