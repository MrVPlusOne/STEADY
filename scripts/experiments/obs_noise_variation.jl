using DataFrames, CSV

my_include = include # to avoid mess up the VSCode linter

perf_list = []

let ° = π / 180
    for obs_w in [0.1, 0.4, 1.0], schedule in [false]
    # for obs_w in [0.1], schedule in [true, false]
    # for σ_bearing in [1°, 2.5°, 5°, 10°, 20°]
        # you can find the available args inside `train_models.jl`.
        global script_args = (;
            gpu_id=7,
            # is_quick_test=true,
            σ_bearing=1°,
            use_obs_weight_schedule=schedule,
            max_obs_weight=obs_w,
            # max_train_steps=20_000,
            exp_name="obs_w=$obs_w, schedule=$(schedule)",
        )
        my_include("../train_models.jl")
        push!(perf_list, Main.perf)
    end
end

result_path = joinpath("results", "obs_schedule_variation_false.csv")
DataFrame(perf_list) |> display
CSV.write(result_path, DataFrame(perf_list))