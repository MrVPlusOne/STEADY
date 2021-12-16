##-----------------------------------------------------------
using Distributions
using StatsPlots
using DrWatson
import Random

quick_test=true
alg_name=:neural  # either :neural or :sindy

StatsPlots.default(dpi=300, legend=:outerbottom)

# generate data
Random.seed!(123)
landmarks = @SVector[@SVector[-1.0, 2.5], @SVector[6.0, -4.0], 
    @SVector[6.0, 12.0], @SVector[10.0, 2.0]]
lInfo = LandmarkInfo(; landmarks, bearing_only=Val(false))
front_drive=true
scenario = Car2dScenario(lInfo, BicycleCarDyn(; front_drive))

times = quick_test ? collect(0.0:0.1:2.0) : collect(0.0:0.1:15)
obs_frames = 1:5:length(times)

params_guess = nothing

n_runs = 10
n_test_runs = 10
n_fit_trajs = method === :neural ? 100 : 10
train_split = 6

train_setups = map(1:n_runs) do i
    x0 = (
        pos=@SVector[-6.5+randn(), 1.2+randn()], 
        vel=@SVector[0.25, 0.0],
        θ=randn()°, 
        ω=0.1randn(),
    )
    ScenarioSetup(times, obs_frames, x0, manual_control(front_drive, 0.2))
end

test_setups = map(1:n_test_runs) do i
    x0 = (
        pos=@SVector[-2.5+2randn(), 2randn()], 
        vel=@SVector[0.25randn(), 0.0],
        θ=randn()°, 
        ω=0.3randn(),
    )
    ScenarioSetup(times, obs_frames, x0, manual_control(front_drive, 0.6))
end
nothing
##-----------------------------------------------------------
# simulate the scenario
save_dir=data_dir("sims", savename("car2d", (; alg_name, quick_test)))
old_motion_model = let 
    sketch=dynamics_sketch(scenario) 
    core=dynamics_core(scenario)
    to_p_motion_model(core, sketch)(true_params)
end

sim_result = simulate_scenario(scenario, old_motion_model, train_setups; save_dir)
nothing
##-----------------------------------------------------------
# test fitting the trajectories
sketch = sindy_sketch(scenario)
algorithm = mk_regressor(alg_name)

post_sampler = ParticleFilterSampler(
    n_particles=quick_test ? 2000 : 50_000,
    n_trajs=100,
)

em_result = let
    comps_σ = [0.5,0.5,0.5]
    dyn_guess = GaussianGenerator(
        _ -> (loc_ax=0.0, loc_ay=0.0, der_ω=0.0),
        (loc_ax=0.1, loc_ay=0.1, der_ω=0.1),
        (μ_f = "all zeros",),
    )
    true_post_trajs = test_posterior_sampling(
        scenario, old_motion_model, "test_truth", 
        sim_result, post_sampler, state_L2_loss=L2_in_SE2).post_trajs
    dyn_est = test_dynamics_fitting(
        scenario, train_split, true_post_trajs, sim_result.obs_data_list, 
        algorithm, sketch, comps_σ, n_fit_trajs)
    (; dyn_est)
    # synthesize_scenario(
    #     scenario, train_split, sim_result, algorithm, sketch, dyn_guess; 
    #     post_sampler, n_fit_trajs, max_iters=quick_test ? 5 : 501)
end
nothing
##-----------------------------------------------------------
# test found dynamics
display(em_result.dyn_est)
analyze_motion_model_performance(
    scenario, em_result.dyn_est, old_motion_model, 
    get_simplified_motion_model(scenario, true_params), test_setups;
    save_dir, post_sampler, state_L2_loss=L2_in_SE2, n_repeat=5,
)
##-----------------------------------------------------------