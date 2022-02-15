include("experiment_common.jl")
using CSV
using DataFrames
SEDL.should_check_finite[] = false

let
    result_name = "EM-real"
    result_dir = joinpath("results/comparisons/$result_name")
    if isfile(result_dir)
        error("\"$result_dir\" already exists")
    end
    mkpath(result_dir)

    perf_best = []
    perf_measure = []

    for train_method in [:Handwritten, :Super_Hand, :Super_TV, :Super_noiseless, :EM]#, :VI]
        # you can find the available args inside `train_models.jl`.
        local script_args = (;
            is_quick_test=true,
            # scenario=SEDL.HovercraftScenario(),
            validation_metric=:RMSE,
            # n_train_ex=256,
            gpu_id=Main.GPU_ID, # set this in the REPL before running the script
            # use_fixed_variance=true,
            # use_simple_obs_model=true,
            train_method,
        )
        local perfs = run_multiple_times(script_args, 4).test_performance
        local measure = map(SEDL.to_measurement, perfs)
        local best = (; log_obs=maximum(perfs.log_obs), RMSE=minimum(perfs.RMSE))
        push!(perf_measure, merge((; method=train_method), measure))
        push!(perf_best, merge((; method=train_method), best))
    end

    println("Saving results to $result_dir")
    table_best = DataFrame(perf_best)
    println("--------------------------------------------\nBest performance:")
    table_best|> display
    CSV.write(joinpath(result_dir, "best.csv"), table_best)
    table_avg = DataFrame(perf_measure)
    println("Average performance:")
    table_avg|> display
    CSV.write(joinpath(result_dir, "average.csv"), table_avg)
end