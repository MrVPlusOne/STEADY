##-----------------------------------------------------------
using Distributions
using StatsPlots
using DrWatson
using Random: Random

StatsPlots.default(; dpi=300, legend=:outerbottom)

# generate data
Random.seed!(123)
landmarks = @SVector([
    @SVector([-1.0, 2.5]),
    @SVector([1.0, -1.0]),
    @SVector([8.0, -5.5]),
    @SVector([14.0, 6.0]),
    @SVector([16.0, -7.5])
])
scenario = HovercraftScenario(LandmarkInfo(; landmarks))
times = collect(0.0:0.1:14)
obs_frames = 1:4:length(times)
true_params = simulation_params(scenario)
params_guess = (;
    mass=1.65,
    drag_x=0.04,
    drag_y=0.1,
    rot_mass=1.66,
    rot_drag=0.09,
    sep=0.87,
    σ_v=0.1,
    σ_ω=0.1,
)

n_runs = 10
n_test_runs = 10
n_fit_trajs = 10
train_split = 6

train_setups, test_setups = let
    setups = map(1:(n_runs + n_test_runs)) do i
        x0 = (
            pos=@SVector([0.5 + randn(), 0.5 + randn()]),
            vel=@SVector([0.25 + 0.3randn(), 0.0 + 0.2randn()]),
            θ=randn()°,
            ω=0.2randn(),
        )
        ScenarioSetup(times, obs_frames, x0, manual_control())
    end
    setups[1:n_runs], setups[(n_runs + 1):(n_runs + n_test_runs)]
end
nothing
##-----------------------------------------------------------
# simulate the scenario
save_dir = data_dir("sims/hovercraft")
old_motion_model = let
    sketch = dynamics_sketch(scenario)
    core = dynamics_core(scenario)
    @show core
    to_p_motion_model(core, sketch)(true_params)
end
println("Ground truth motion model:")
display(sindy_core(scenario, true_params))

new_motion_model = let
    sketch = sindy_sketch(scenario)
    core = sindy_core(scenario, true_params)
    GaussianMotionModel(sketch, core)
end
sim_result = simulate_scenario(scenario, new_motion_model, train_setups; save_dir)
nothing
##-----------------------------------------------------------
# test fitting the trajectories
sketch = sindy_sketch(scenario)
algorithm = let
    shape_env = ℝenv()
    comp_env = ComponentEnv()
    components_scalar_arithmatic!(comp_env; can_grow=true)

    basis_expr = TAST[]
    for v1 in sketch.input_vars
        push!(basis_expr, v1)
        for v2 in sketch.input_vars
            if v2.name <= v1.name
                push!(basis_expr, v1 * v2)
            end
        end
    end
    @show basis_expr
    basis = [compile(e, shape_env, comp_env) for e in basis_expr]
    lambdas = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8] .* 5
    @unzip optimizer_list, optimizer_descs = map(lambdas) do λ
        regressor = LassoRegression(λ; fit_intercept=true)
        GeneralizedLinearRegression()
        # regressor = RidgeRegression(10.0; fit_intercept=false)
        SeqThresholdOptimizer(0.1, regressor), (λ=λ,)
    end

    SindyRegression(comp_env, basis, optimizer_list, optimizer_descs)
end

post_sampler = ParticleFilterSampler(; n_particles=60_000, n_trajs=100)

em_result = let
    comps_σ = [0.1, 0.1, 0.1]
    comps_guess = GaussianGenerator(
        _ -> (loc_ax=0.0, loc_ay=0.0, der_ω=0.0),
        (loc_ax=0.1, loc_ay=0.1, der_ω=0.1),
        (μ_f="all_zeros",),
    )
    true_post_trajs =
        test_posterior_sampling(
            scenario,
            old_motion_model,
            "test_truth",
            sim_result,
            post_sampler;
            state_L2_loss=L2_in_SE2,
        ).post_trajs
    test_dynamics_fitting(
        scenario,
        train_split,
        true_post_trajs,
        sim_result.obs_data_list,
        algorithm,
        sketch,
        comps_σ,
        n_fit_trajs,
    )
    synthesize_scenario(
        scenario,
        train_split,
        sim_result,
        algorithm,
        sketch,
        comps_guess;
        post_sampler,
        n_fit_trajs,
        max_iters=501,
    )
end
nothing
##-----------------------------------------------------------
# test found dynamics
analyze_motion_model_performance(
    scenario,
    em_result.dyn_est,
    old_motion_model,
    get_simplified_motion_model(scenario, true_params),
    test_setups;
    save_dir,
    post_sampler,
    state_L2_loss=L2_in_SE2,
)
##-----------------------------------------------------------
