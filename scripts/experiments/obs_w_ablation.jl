include("experiment_common.jl")

scenarios = [
    # SEDL.RealCarScenario("ut_automata"),
    # SEDL.RealCarScenario("alpha_truck"),
    SEDL.HovercraftScenario(),
]

for sce in scenarios
    with_alert("obs_w_ablation.jl [scenario=$sce]") do
        out_dir = mkpath(joinpath("results/obs_w_ablation/"))

        local script_args = (;
            # is_quick_test=true,
            scenario=sce,
            train_method=:EM,
            gpu_id=Main.GPU_ID,
            use_obs_weight_schedule=false,
            n_train_ex=160, 
        )

        perfs = train_multiple_times(script_args, 3).test_performance
        measure = map(SEDL.to_measurement, perfs)

        write(joinpath(out_dir, "$sce-real-160.txt"), string(measure))
        write(joinpath(out_dir, "$sce-real-160-perfs.txt"), string(perfs))
    end
end