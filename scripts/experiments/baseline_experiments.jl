include("experiment_common.jl")
using CSV
using DataFrames
SEDL.should_check_finite[] = false

with_alert("baseline_experiments.jl") do
    # result_name = "hovercraft"
    result_name = "alpha_truck-SVI"
    group_name = "comparisons"

    result_dir = joinpath("reports/$group_name/$result_name")
    if ispath(result_dir)
        error("\"$result_dir\" already exists")
    end
    mkpath(result_dir)

    perf_best = []
    perf_measure = []

    for train_method in [:SVI] # [:Handwritten, :FitHand, :FitTV, :FitTruth, :EM, :SVI]
        # you can find the available args inside `train_models.jl`.
        local script_args = (;
            # is_quick_test=true,
            # scenario=SEDL.HovercraftScenario(16),
            scenario=SEDL.RealCarScenario("alpha_truck"),
            gpu_id=Main.GPU_ID, # set this in the REPL before running the script
            exp_group=group_name,
            # :log_obs seeem to be a poor metric for :SVI
            validation_metric=(train_method == :SVI ? :RMSE : :log_obs),
            train_method,
        )
        local perfs = train_multiple_times(script_args, 3).test_performance

        local measure = map(SEDL.to_measurement, perfs)
        local max_metrics = (; log_obs=maximum(perfs.log_obs))
        local min_metrics = map(minimum, SEDL.dropnames(perfs, (:log_obs,)))
        local best = merge(max_metrics, min_metrics)

        push!(perf_measure, merge((; method=train_method), measure))
        push!(perf_best, merge((; method=train_method), best))

        CSV.write(joinpath(result_dir, "best.csv"), DataFrame(perf_best))
        CSV.write(joinpath(result_dir, "average.csv"), DataFrame(perf_measure))
    end

    println("----------------------------------------------")
    println("Best performance:")
    display(DataFrame(perf_best))

    println("----------------------------------------------")
    println("Average performance:")
    display(DataFrame(perf_measure))

    @info("Results saved to $result_dir ...")
end