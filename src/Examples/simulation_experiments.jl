@kwdef struct ExperimentConfigs
    n_train_setups::Int
    n_test_setups::Int
    regress_alg::Symbol
    is_test_run::Bool
end

function run_experiment(sce::Scenario, configs::ExperimentConfigs, save_dir)
    (; n_train_setups, n_test_setups, regress_alg, is_test_run) = configs

    dyn_params = simulation_params(sce)
    true_motion_model = let 
        sketch=dynamics_sketch(sce) 
        core=dynamics_core(sce)
        to_p_motion_model(core, sketch)(dyn_params)
    end


    # generate simulation data
    train_setups = map(1:n_train_setups) do _
        times = collect(is_test_run ? (0.0:0.1:2.0) : (0.0:0.1:10))
        ScenarioSetup(;
            times,
            obs_frames = 1:length(times),
            x0 = simulation_x0(sce),
            controller = simulation_controller(sce),
        )
    end
    train_sim = simulate_scenario(
        sce, true_motion_model, train_setups; save_dir=joinpath(save_dir, "train")
    )

    test_setups = map(1:n_test_setups) do _
        times = collect(is_test_run ? (0.0:0.1:3.0) : (0.0:0.1:15))
        ScenarioSetup(;
            times,
            obs_frames = 1:5:length(times),
            x0 = simulation_x0(sce),
            controller = simulation_controller(sce),
        )
    end
    test_sim = simulate_scenario(
        sce, true_motion_model, test_setups; save_dir=joinpath(save_dir, "test")
    )

    post_sampler = ParticleFilterSampler(;
        n_particles=is_test_run ? 2000 : 50_000, n_trajs=100
    )

    true_post_trajs =
        test_posterior_sampling(
            sce,
            true_motion_model,
            "test_truth",
            train_sim,
            post_sampler;
            state_L2_loss=state_L2_loss(sce),
        ).post_trajs

    train_split = ceil(Int, n_train_setups * 0.6)
    sketch = sindy_sketch(sce)
    comps_σ = fill(1.0, length(sketch.output_vars))
    dyn_guess = let
        init_μ = NamedTuple(v.name => 0.0 for v in sketch.output_vars)
        init_σ = NamedTuple(v.name => 1.0 for v in sketch.output_vars)
        GaussianGenerator(_ -> init_μ, init_σ, (μ_f="all zeros",))
    end

    n_fit_trajs = method === :neural ? 100 : 20
    # learning from the true posterior
    dyn_from_posterior = test_dynamics_fitting(
        sce,
        train_split,
        true_post_trajs,
        train_sim.obs_data_list,
        mk_regressor(regress_alg, sketch),
        sketch,
        comps_σ,
        n_fit_trajs,
    )

    return analyze_motion_model_performance(
        sce,
        dyn_from_posterior,
        true_motion_model,
        get_simplified_motion_model(sce, dyn_params),
        test_sim;
        save_dir=joinpath(save_dir, "test"),
        post_sampler,
        state_L2_loss=L2_in_SE2,
        n_repeat=5,
    )
end

function run_simulation_experiments(; is_test_run)
    exp_dir = data_dir(is_test_run ? "batch_experiments-test_run" : "batch_experiments")

    landmarks = @SVector([
        @SVector([-1.0, 2.5]),
        @SVector([6.0, -4.0]),
        @SVector([6.0, 12.0]),
        @SVector([10.0, 2.0]),
    ])
    lInfo = LandmarkInfo(; landmarks, bearing_only=Val(false))
    scenarios = [
        "car2d-front_drive" => Car2dScenario(lInfo, BicycleCarDyn(; front_drive=true)),
        "car2d-rear_drive" => Car2dScenario(lInfo, BicycleCarDyn(; front_drive=false)),
    ]


    for (name, sce) in scenarios, alg_name in [:sindy, :neural]
        configs = ExperimentConfigs(;
            n_train_setups=10,
            n_test_setups=50,
            regress_alg=alg_name,
            is_test_run=is_test_run,
        )
        save_dir = joinpath(exp_dir, savename(name, (; alg_name)))
        run_experiment(sce, configs, save_dir)
    end
end