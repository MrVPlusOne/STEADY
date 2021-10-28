##-----------------------------------------------------------
using Distributions
using StatsPlots
import Random

StatsPlots.default(dpi=600, legend=:outerbottom)
##-----------------------------------------------------------
# generate data
Random.seed!(123)
times = collect(0.0:0.1:12)
obs_frames = 1:5:length(times)
# true_params = (; l1 = 0.5, τ1 = 0.8, v1=0.1, a1=1.0)
true_params = (; l1 = 0.5, τ1 = 0.8, σ_θ=0.1, σ_v=0.2)
true_σ = (; σ_pos=@SVector[0.04, 0.04])
tiny_σ = (; σ_pos=@SVector[0.01, 0.01], σ_v=@SVector[0.01, 0.01])

car_model = Car2D.simple_model(true_σ)
(; sketch, sketch_core) = car_model
true_motion_model = to_p_motion_model(sketch_core, sketch)(true_params)

vdata = Car2D.variable_data()
x0_dist = init_state_distribution(vdata)

landmarks = @SVector[@SVector[-1.0, 2.5], @SVector[6.0, -4.0], @SVector[-0.6, -1.0]]
noise_scale = 1.0
true_system = MarkovSystem(x0_dist, true_motion_model, 
    Car2D.sensor_dist(landmarks; noise_scale))

est_params = (; l1 = 0.75, τ1 = 0.5, σ_θ=0.4, σ_v=0.4)
est_motion_model = to_p_motion_model(sketch_core, sketch)(est_params)
est_system = MarkovSystem(x0_dist, est_motion_model, Car2D.sensor_dist(landmarks; noise_scale))

x0 = (pos=@SVector[0.5, 0.5], θ=-5°, v=@SVector[0.25, 0.0])
ex_data = simulate_trajectory(times, x0, true_system, Car2D.manual_control())
obs_data = (; times, obs_frames, ex_data.observations, ex_data.controls)
@df ex_data.controls plot(times, [:v̂, :ϕ̂], label=["v̂" "ϕ̂"])
let traj_p = plot(legend=:outerbottom, aspect_ratio=1.0)
    Car2D.plot_states!(ex_data.states, "truth"; landmarks, obs_data)
end |> display
log_score_float = (d,x) -> log_score(d, x, Float64)
##-----------------------------------------------------------
# sample posterior using the correct dynamics
# ffbs_result = @time ffbs_smoother(true_system, obs_data, n_particles=50_000, n_trajs=100)
ffbs_trajs = @time ThreadPools.bmap(1:9) do i 
    r = filter_smoother(true_system, obs_data, 
        n_particles=50_000, n_trajs=55, showprogress=i==1)
    r.trajectories
end |> ts -> reduce(vcat, ts)
# map_result = @time map_trajectory(
#     true_system, obs_data, ffbs_result.trajectories[2, :]; 
#     optim_options=Optim.Options(f_abstol=1e-4, iterations=1000),    
# )
let plt = plot()
    Car2D.plot_trajectories!(ffbs_trajs, linealpha=0.04, linecolor=2)
    Car2D.plot_states!(ex_data.states, "truth"; landmarks, obs_data, 
        state_color=1, landmark_color=3)
    # Car2D.plot_states!(map_result.states, "MAP states"; landmarks=[], obs_data, 
    #     state_color=:red, state_alpha=0.6)
    display("image/png", plt)
end
##-----------------------------------------------------------
# test the PGAS sampler
init_traj = ffbs_trajs[1, :]
pgas_trajs = ThreadPools.bmap(1:9) do i
    r = PGAS_smoother(est_system, obs_data, init_traj,
        n_particles=500, n_trajs=55, n_thining=2, showprogress=i==1)
    r.trajectories
end |> ts -> reduce(vcat, ts)
let plt = plot()
    Car2D.plot_trajectories!(pgas_trajs, linealpha=0.04, linecolor=2)
    Car2D.plot_states!(ex_data.states, "truth"; landmarks, obs_data, 
        state_color=1, landmark_color=3)
    display("image/png", plt)
end
##-----------------------------------------------------------
# simple fitting test
params_dist = params_distribution(sketch)
check_params_logp(ex_data.states[1], x0_dist, true_params, params_dist)
@time fit_dynamics_params(
    WrappedFunc(to_p_motion_model(sketch_core, sketch)),
    ffbs_trajs,
    x0_dist, params_dist,
    obs_data,
    rand(params_dist),
    optim_options = Optim.Options(f_abstol=1e-4),
)
##-----------------------------------------------------------
# program enumeration
shape_env = ℝenv()
comp_env = ComponentEnv()
components_scalar_arithmatic!(comp_env, can_grow=true)
components_special_functions!(comp_env, can_grow=true)

pruner = IOPruner(; inputs=sample_rand_inputs(sketch, 100), comp_env)
senum = @time synthesis_enumeration(vdata, sketch, shape_env, comp_env, 6; pruner)
let rows = map(senum.enum_result.pruned) do r
        (; r.pruned, r.reason)
    end
    println("---- Pruned programs ----")
    show(DataFrame(rows), truncate=100)
    println()
end
display(senum)
##-----------------------------------------------------------
# run full synthesis
optim_options = Optim.Options(
    f_abstol=1e-4,
    iterations=100,
    time_limit=10.0,
)
fit_settings = DynamicsFittingSettings(; 
    optim_options, evals_per_program=2, n_threads=10)

function iter_callback((; iter, trajectories, dyn_est))
    show_MAP = false
    if iter <= 10 || mod1(iter, 5) == 1
        if show_MAP
            @info "Computing trajectory MAP estimation..."
            local motion_model = dyn_est.p_motion_model(dyn_est.params)
            local system_est = MarkovSystem(x0_dist, motion_model, true_system.obs_model)
            local best_traj = collect(get_rows(trajectories)) |> 
                max_by(traj -> total_log_score(system_est, obs_data, traj, Float64))
            local map_result = @time map_trajectory(
                system_est, obs_data, best_traj; 
                optim_options=Optim.Options(f_abstol=1e-4, iterations=1000),    
            )
        end

        p = plot(title="iteration $iter")
        Car2D.plot_trajectories!(trajectories, linealpha=0.15, linecolor=2)
        Car2D.plot_states!(ex_data.states, "truth"; landmarks, obs_data, 
            state_color=1, landmark_color=3)
        show_MAP && Car2D.plot_states!(map_result.states, "MAP states"; landmarks, obs_data, 
            state_color=:red, state_alpha=0.6)
        display("image/png", p)
    end
end

iter_result = let 
    comps_guess = let v̂ = Car2D.U_Speed, θ = Car2D.Orientation, τ1 = Var(:τ1, ℝ, PUnits.Time)
        f_zero = get_component(comp_env, :zero)
        (der_vx=f_zero(v̂)/τ1, der_θ = f_zero(θ)/τ1,)
    end

    @time fit_dynamics_iterative(senum, obs_data, comps_guess, est_params;
        obs_model=true_system.obs_model,
        program_logp=prog_size_prior(0.2), fit_settings,
        max_iters=101,
        iteration_callback = iter_callback,
    )
end
display(iter_result)
plot(iter_result, start_idx=10)
##-----------------------------------------------------------