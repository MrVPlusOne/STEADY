@kwdef struct ExperimentConfigs
    n_train_setups::Int
    n_test_setups::Int
    regress_algs::Vector{Symbol}
    is_test_run::Bool
end

function run_experiment(sce::Scenario, configs::ExperimentConfigs, save_dir, timer)
    (; n_train_setups, n_test_setups, regress_algs, is_test_run) = configs

    dyn_params = simulation_params(sce)
    true_motion_model = let
        sketch = dynamics_sketch(sce)
        core = dynamics_core(sce)
        to_p_motion_model(core, sketch)(dyn_params)
    end


    # generate simulation data
    @timeit timer "simulation" begin
        train_setups = map(1:n_train_setups) do _
            times = collect(is_test_run ? (0.0:0.1:2.0) : (0.0:0.1:10))
            ScenarioSetup(;
                times,
                obs_frames=1:4:length(times),
                x0=simulation_x0(sce),
                controller=simulation_controller(sce),
            )
        end
        train_sim = simulate_scenario(
            sce, true_motion_model, train_setups; save_dir=joinpath(save_dir, "train")
        )

        test_setups = map(1:n_test_setups) do _
            times = collect(is_test_run ? (0.0:0.1:3.0) : (0.0:0.1:15))
            ScenarioSetup(;
                times,
                obs_frames=1:5:length(times),
                x0=simulation_x0(sce),
                controller=simulation_controller(sce),
            )
        end
        test_sim = simulate_scenario(
            sce, true_motion_model, test_setups; save_dir=joinpath(save_dir, "test")
        )
    end

    post_sampler = ParticleFilterSampler(;
        n_particles=is_test_run ? 2000 : 50_000, n_trajs=100
    )

    true_post_trajs = @timeit timer "training data sampling" begin
        test_posterior_sampling(
            sce,
            true_motion_model,
            "true_model",
            train_sim,
            post_sampler;
            state_L2_loss=state_L2_loss(sce),
        ).post_trajs
    end

    train_split = ceil(Int, n_train_setups * 0.6)
    sketch = sindy_sketch(sce)
    comps_σ = fill(1.0, length(sketch.output_vars))
    dyn_guess = let
        init_μ = NamedTuple(v.name => 0.0 for v in sketch.output_vars)
        init_σ = NamedTuple(v.name => 1.0 for v in sketch.output_vars)
        GaussianGenerator(_ -> init_μ, init_σ, (μ_f="all zeros",))
    end

    # learning from the true posterior
    n_fit_trajs = post_sampler.n_trajs
    models_from_posterior = []
    foreach(regress_algs) do alg
        regressor = mk_regressor(alg, sketch; is_test_run)
        @timeit timer "dynamics_fitting: $alg" begin
            @info("dynamics_fitting: $alg")
            dyn_est = test_dynamics_fitting(
                sce,
                train_split,
                true_post_trajs,
                train_sim.obs_data_list,
                regressor,
                sketch,
                comps_σ,
                n_fit_trajs,
            )
        end
        push!(models_from_posterior, string(alg) => GaussianMotionModel(sketch, dyn_est))
    end

    push!(
        models_from_posterior,
        "handcrafted" => get_simplified_motion_model(sce, dyn_params),
        "true_model" => true_motion_model,
    )

    datasets = ["train" => train_sim, "test" => test_sim]

    rows = []
    @withprogress name = "analyze_motion_model_performance" begin
        for (model_name, model) in models_from_posterior
            cols = @timeit timer "analyze_model: $model_name" begin
                map(datasets) do (data_name, data)
                    metrics =
                        test_posterior_sampling(
                            sce,
                            model,
                            "$data_name ($model_name)",
                            data,
                            post_sampler;
                            state_L2_loss=state_L2_loss(sce),
                            generate_plots=true,
                        ).metrics
                    map(collect(metrics)) do (mname, mvalue)
                        Symbol("$mname-$data_name") => mvalue
                    end
                end |> vcatreduce
            end
            push!(rows, (; model_name, cols...))
        end
    end
    metric_table = DataFrame(rows)
    display(metric_table)
    CSV.write(joinpath(save_dir, "metrics.csv"), metric_table)
    @tagsave(joinpath(save_dir, "metrics.jld2"), @strdict(metric_table))

    metric_table
end

function run_simulation_experiments(; is_test_run)
    exp_dir = data_dir(is_test_run ? "test_run-batch_experiments" : "batch_experiments")

    scenarios = [
        "hovercraft" => hovercraft_scenario(),
        "car2d-front_drive" => car2d_scenario(; front_drive=true),
        "car2d-rear_drive" => car2d_scenario(; front_drive=false),
    ]

    configs = ExperimentConfigs(;
        n_train_setups=10,
        n_test_setups=50,
        regress_algs=[:genetic, :sindy_ssr], #[:neural, :neural_skip, :sindy, :sindy_ssr],
        is_test_run=is_test_run,
    )

    timer = TimerOutput()
    results = []
    for (name, sce) in scenarios
        save_dir = joinpath(exp_dir, name)
        @timeit timer "run_experiment($name)" begin
            @info("Running experiment: $name")
            push!(results, name => run_experiment(sce, configs, save_dir, timer))
        end
    end
    TimerOutputs.complement!(timer)
    display(timer)
    results
end