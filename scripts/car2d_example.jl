##-----------------------------------------------------------
using Distributions
using StatsPlots
import Random

StatsPlots.default(dpi=600, legend=:outerbottom)
##-----------------------------------------------------------
# generate data
Random.seed!(123)
times = collect(0.0:0.1:10)
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

landmarks = @SVector[@SVector[1.0, 2.5], #=@SVector[6.0, 1.5],=# @SVector[4.0, -2.0]]
noise_scale = 1.0
true_system = MarkovSystem(x0_dist, true_motion_model, 
    Car2D.sensor_dist(landmarks, noise_scale))

x0 = (pos=@SVector[0.5, 0.5], θ=-5°, v=@SVector[0.25, 0.0])
ex_data = simulate_trajectory(times, x0, true_system, Car2D.manual_control())
obs_data = (; times, obs_frames, ex_data.observations, ex_data.controls)
@df ex_data.controls plot(times, [:v̂, :ϕ̂], label=["v̂" "ϕ̂"])
let traj_p = plot(legend=:outerbottom, aspect_ratio=1.0)
    Car2D.plot_states!(ex_data.states, "truth"; landmarks, obs_frames)
end |> display
log_score_float = (d,x) -> log_score(d, x, Float64)
##-----------------------------------------------------------
# sample posterior using the correct dynamics
# ffbs_result = @time ffbs_smoother(true_system, obs_data, n_particles=10_000, n_trajs=100)
ffbs_result = @time filter_smoother(true_system, obs_data, n_particles=500_000, n_trajs=100)
let plt = plot()
    Car2D.plot_trajectories!(ffbs_result.trajectories)
    Car2D.plot_states!(ex_data.states, "truth"; landmarks, obs_frames)
    display("image/png", plt)
end
##-----------------------------------------------------------
# simple fitting test
params_dist = params_distribution(sketch)
check_params_logp(ex_data.states[1], x0_dist, true_params, params_dist)
@time fit_dynamics_params(
    WrappedFunc(to_p_motion_model(sketch_core, sketch)),
    ffbs_result.trajectories,
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
senum = @time synthesis_enumeration(vdata, sketch, shape_env, comp_env, 5; pruner)
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

iter_result = let 
    comps_guess = let v̂ = Car2D.U_Speed, θ = Car2D.Orientation, τ1 = Var(:τ1, ℝ, PUnits.Time)
        f_zero = get_component(comp_env, :zero)
        (der_vx=f_zero(v̂)/τ1, der_θ = f_zero(θ)/τ1,)
    end
    params_guess = nothing # (mass=3.0, drag=0.8,)

    @time fit_dynamics_iterative(senum, obs_data, comps_guess, params_guess;
        obs_model=true_system.obs_model,
        program_logp=prog_size_prior(0.2), fit_settings,
        max_iters=101,
        iteration_callback = (iter, particles) -> let
            if iter <= 10 || mod1(iter, 5) == 1
                p = plot(title="iteration $iter")
                Car2D.plot_trajectories!(particles)
                Car2D.plot_states!(ex_data.states, "truth"; landmarks, obs_frames)
                display("image/png", p)
            end
        end
    )
end
display(iter_result)
plot(iter_result, start_idx=5)
##-----------------------------------------------------------