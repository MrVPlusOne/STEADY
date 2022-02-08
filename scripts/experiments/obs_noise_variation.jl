using DataFrames, CSV

my_include = include # to avoid mess up the VSCode linter

perf_list = []

schedule = false
σ_bearing_deg = 1.0

let ° = π / 180
    for obs_w in [0.25, 4.0]
    # for σ_bearing in [1°, 2.5°, 5°, 10°, 20°]
        # you can find the available args inside `train_models.jl`.
        global script_args = (;
            gpu_id=Main.GPU_ID, # set this in the REPL before running the script
            # is_quick_test=true,
            σ_bearing=σ_bearing_deg * °,
            use_obs_weight_schedule=schedule,
            max_obs_weight=obs_w,
            max_train_steps=40_000,
            exp_name="obs_w=$obs_w, schedule=$(schedule)",
        )
        my_include("../train_models.jl")
        push!(perf_list, Main.test_performance)
    end
end

result_path = joinpath("results", "obs_schedule_variation_$(σ_bearing_deg)_$schedule.csv")
DataFrame(perf_list) |> display
CSV.write(result_path, DataFrame(perf_list))