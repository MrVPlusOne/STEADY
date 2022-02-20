include("experiment_common.jl")
using CSV
using DataFrames
SEDL.should_check_finite[] = false

with_alert("baseline_experiments.jl") do
    result_name = "hovercraft160"

    result_dir = joinpath("results/comparisons/$result_name")
    if ispath(result_dir)
        error("\"$result_dir\" already exists")
    end
    mkpath(result_dir)

    perf_best = []
    perf_measure = []

    for train_method in [:Handwritten, :FitHand, :FitTV, :FitTruth, :EM, :SVI]
        # you can find the available args inside `train_models.jl`.
        local script_args = (;
            # is_quick_test=true,
            scenario=SEDL.HovercraftScenario(),
            # scenario=SEDL.RealCarScenario("ut_automata"),
            # scenario=SEDL.RealCarScenario("alpha_truck"),
            n_train_ex=160,
            gpu_id=Main.GPU_ID, # set this in the REPL before running the script
            train_method,
        )
        local perfs = train_multiple_times(script_args, 3).test_performance

        local measure = map(SEDL.to_measurement, perfs)
        local best = (; log_obs=maximum(perfs.log_obs), RMSE=minimum(perfs.RMSE))
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