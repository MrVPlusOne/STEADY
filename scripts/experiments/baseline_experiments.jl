include("experiment_common.jl")
using CSV
using DataFrames
SEDL.should_check_finite[] = false

let
    result_name = "small_car"
    result_dir = joinpath("results/comparisons/$result_name")
    if isfile(result_dir)
        error("\"$result_dir\" already exists")
    end
    mkpath(result_dir)

    perf_best = []
    perf_measure = []

    for train_method in [:Handwritten, :FitHand, :FitTV, :FitTruth, :EM, :SVI]
        # you can find the available args inside `train_models.jl`.
        local script_args = (;
            # is_quick_test=true,
            # scenario=SEDL.HovercraftScenario(),
            scenario=SEDL.RealCarScenario("ut_automata"),
            validation_metric=:RMSE,
            # n_train_ex=256,
            gpu_id=Main.GPU_ID, # set this in the REPL before running the script
            # use_fixed_variance=true,
            # use_simple_obs_model=true,
            train_method,
        )
        local perfs = train_multiple_times(script_args, 3).test_performance

        local measure = map(SEDL.to_measurement, perfs)
        local best = (; log_obs=maximum(perfs.log_obs), RMSE=minimum(perfs.RMSE))
        push!(perf_measure, merge((; method=train_method), measure))
        push!(perf_best, merge((; method=train_method), best))
    end

    @info("Saving results to $result_dir ...")
    table_best = DataFrame(perf_best)
    println("----------------------------------------------")
    println("Best performance:")
    table_best|> display
    CSV.write(joinpath(result_dir, "best.csv"), table_best)
    table_avg = DataFrame(perf_measure)
    println("----------------------------------------------")
    println("Average performance:")
    table_avg|> display
    CSV.write(joinpath(result_dir, "average.csv"), table_avg)

    alert("Baseline experiments finished: $result_name. (GPU ID: $(Main.GPU_ID))")
end