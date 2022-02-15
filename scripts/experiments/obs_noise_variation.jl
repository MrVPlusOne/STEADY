using SEDL
using CSV
using DataFrames
SEDL.should_check_finite[] = false

my_include = include # to avoid mess up the VSCode linter

σ_deg=12
result_name = "$(σ_deg)°"
println("Starting experiment: $result_name...")
result_path = joinpath("results/vary_obs_noise", "$result_name.csv")
mkpath(dirname(result_path))
if isfile(result_path)
    error("file $result_path already exists")
end

perf_list = []

for train_method in [:Handwritten, :FitHand, :FitTv, :FitTruth, :EM, :SVI]
    # you can find the available args inside `train_models.jl`.
    global script_args = (;
        # is_quick_test=true,
        # scenario=SEDL.HovercraftScenario(),
        validation_metric=:RMSE,
        # n_train_ex=256,
        gpu_id=Main.GPU_ID, # set this in the REPL before running the script
        # use_fixed_variance=true,
        # use_simple_obs_model=true,
        σ_bearing=σ_deg * °,
        train_method,
    )
    my_include("../train_models.jl")
    push!(perf_list, Main.test_performance)
end

results = DataFrame(perf_list)
results|> display
CSV.write(result_path, results)