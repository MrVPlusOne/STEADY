include("experiment_common.jl")

with_alert("obs_w_ablation.jl") do
    scenarios = [
        SEDL.HovercraftScenario(),
        SEDL.RealCarScenario("ut_automata"),
        SEDL.RealCarScenario("alpha_truck"),
    ]

    out_dir = mkpath(joinpath("results/obs_w_ablation/"))

    for sce in scenarios
        local script_args = (;
            scenario=sce,
            train_method=:EM,
            gpu_id=Main.GPU_ID,
            use_obs_weight_schedule=false,
        )

        perfs = train_multiple_times(script_args, 3).test_performance
        measure = map(SEDL.to_measurement, perfs)

        write(joinpath(out_dir, string(sce)), string(measure))
    end
end