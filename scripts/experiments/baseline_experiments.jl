using SEDL
using CSV
using DataFrames
SEDL.should_check_finite[] = false

my_include = include # to avoid mess up the VSCode linter

result_name = "real-S"
mkpath("results/comparisons")
result_path = joinpath("results/comparisons", "$result_name.csv")
if isfile(result_path)
    error("file $result_path already exists")
end

perf_list = []

for train_method in [:Handwritten, :Super_Hand, :Super_TV, :Super_noiseless, :EM, :VI]
    # you can find the available args inside `train_models.jl`.
    global script_args = (;
        # is_quick_test=true,
        # scenario=SEDL.HovercraftScenario(),
        validation_metric=:RMSE,
        # n_train_ex=256,
        gpu_id=Main.GPU_ID, # set this in the REPL before running the script
        # use_fixed_variance=true,
        use_simple_obs_model=true,
        train_method,
    )
    my_include("../train_models.jl")
    push!(perf_list, Main.test_performance)
end

results = DataFrame(perf_list)
results|> display
CSV.write(result_path, results)