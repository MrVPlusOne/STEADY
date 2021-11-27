##-----------------------------------------------------------
using Distributions
using StatsPlots
using DrWatson
import Random

StatsPlots.default(dpi=300, legend=:outerbottom)

# generate data
Random.seed!(123)
landmarks = @SVector[@SVector[-1.0, 2.5], @SVector[6.0, -4.0], 
    @SVector[6.0, 12.0], @SVector[10.0, 2.0]]
lInfo = LandmarkInfo(; landmarks, bearing_only=Val(false))
front_drive=true
scenario = Car2dScenario(lInfo, BicycleCarDyn(; front_drive))

times = collect(0.0:0.1:15)
obs_frames = 1:5:length(times)

true_params = (; 
    mass=2.0, drag_x=0.05, drag_y=0.11, rot_mass=0.65, rot_drag=0.07,
    sep=0.48, len=0.42, fraction_max=1.5, σ_v=0.011, σ_ω=0.008,)
params_guess = nothing

function manual_control(front_drive)
    pert(x) = x + 0.01randn()
    @unzip times, v̂_seq, steer_seq = if front_drive
        [
            (t=0.0, v̂=0.0, steer=0.0),
            (t=1.0, v̂=pert(3.2), steer=pert(10°)),
            (t=4.0, v̂=pert(3.0), steer=pert(10°)),
            (t=4.5, v̂=pert(3.3), steer=pert(-30°)),
            (t=6.5, v̂=pert(3.3), steer=pert(-30°)),
            (t=7.2, v̂=pert(2.0), steer=pert(20°)),
            (t=9.0, v̂=pert(1.8), steer=pert(20°)),
            (t=9.6, v̂=pert(2.5), steer=pert(10°)),
            (t=15.0, v̂=2.0, steer=0.0),
        ]
    else
        [
            (t=0.0, v̂=0.0, steer=0.0),
            (t=1.0, v̂=pert(2.0), steer=pert(0°)),
            (t=4.0, v̂=pert(2.0), steer=pert(0°)),
            (t=4.5, v̂=pert(2.5), steer=pert(-20°)),
            (t=6.5, v̂=pert(2.9), steer=pert(-20°)),
            (t=7.2, v̂=pert(3.2), steer=pert(20°)),
            (t=9.0, v̂=pert(3.0), steer=pert(10°)),
            (t=10.6, v̂=pert(2.4), steer=pert(0°)),
            (t=15.0, v̂=2.0, steer=0.0),
        ]
    end
    if rand() < 0.6
        steer_seq = -steer_seq
    end
    v̂_f = LinearInterpolation(times, v̂_seq)
    steer_f = LinearInterpolation(times, steer_seq)
    (s, obs, t::Float64) -> begin
        (v̂ = v̂_f(t), steer = steer_f(t))
    end
end

n_runs = 6
n_fit_trajs = 15
n_test_runs = 5
train_setups, test_setups = let 
    setups = map(1:n_runs+n_test_runs) do i
        x0 = (
            pos=@SVector[-6.5+randn(), 1.2+randn()], 
            vel=@SVector[0.25, 0.0],
            θ=randn()°, 
            ω=0.1randn(),
        )
        ScenarioSetup(times, obs_frames, x0, manual_control(front_drive))
    end
    setups[1:n_runs], setups[n_runs+1:n_runs+n_test_runs]
end
nothing
##-----------------------------------------------------------
# simulate the scenario
save_dir=datadir("sims/car2d")
old_motion_model = let 
    sketch=dynamics_sketch(scenario) 
    core=dynamics_core(scenario)
    @show core
    to_p_motion_model(core, sketch)(true_params)
end

sim_result = simulate_scenario(scenario, old_motion_model, train_setups; save_dir)
nothing
##-----------------------------------------------------------
# test fitting the trajectories
algorithm = let
    sketch = sindy_sketch(scenario)
    shape_env = ℝenv()
    comp_env = ComponentEnv()
    components_scalar_arithmatic!(comp_env, can_grow=true)

    basis_expr = TAST[]
    basis_weights = Float64[]
    for v1 in sketch.input_vars
        push!(basis_expr, v1)
    end
    @show basis_expr
    basis = [compile(e, shape_env, comp_env) for e in basis_expr]
    lambdas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]
    @unzip optimizer_list, optimizer_descs = map(lambdas) do λ
        regressor = LassoRegression(λ; fit_intercept=true)
        GeneralizedLinearRegression()
        # regressor = RidgeRegression(10.0; fit_intercept=false)
        SeqThresholdOptimizer(0.1, regressor), (λ=λ,)
    end
    
    SindySynthesis(comp_env, basis, sketch, optimizer_list, optimizer_descs)
end

post_sampler = ParticleFilterSampler(
    n_particles=60_000,
    n_trajs=100,
)

em_result = let
    comps_σ = [0.1,0.1,0.1]
    comps_guess = OrderedDict(
        :loc_ax => GaussianComponent(_ -> 0.0, 0.1),
        :loc_ay => GaussianComponent(_ -> 0.0, 0.1),
        :der_ω => GaussianComponent(_ -> 0.0, 0.1),
    )
    train_split = n_runs ÷ 2
    # true_post_trajs = test_posterior_sampling(
    #     scenario, old_motion_model, "test_truth", sim_result, post_sampler).post_trajs
    # test_dynamics_fitting(
    #     scenario, train_split, true_post_trajs, sim_result.obs_data_list, 
    #     algorithm, comps_σ, n_fit_trajs)
    synthesize_scenario(
        scenario, train_split, sim_result, algorithm, comps_guess; 
        post_sampler, n_fit_trajs)
end
nothing
##-----------------------------------------------------------
# test found dynamics
test_save_dir=datadir("sims/car2d/test")
test_mm = let 
    sketch = sindy_sketch(scenario)
    sindy_motion_model(sketch, NamedTuple(found_comps))
end

test_sim_result = simulate_scenario(scenario, old_motion_model, test_setups; save_dir=test_save_dir)
metrics_trained = test_posterior_sampling(
    scenario, test_mm, "test_trained", test_sim_result, post_sampler).metrics
metrics_truth = test_posterior_sampling(
    scenario, old_motion_model, "test_truth", test_sim_result, post_sampler).metrics
metrics_trained
metrics_truth
##-----------------------------------------------------------