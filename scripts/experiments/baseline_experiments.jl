using SEDL
using CSV
using DataFrames
SEDL.should_check_finite[] = false

my_include = include # to avoid mess up the VSCode linter

result_name = "comparisons_hovercraft"
mkpath("results")
result_path = joinpath("results", "$result_name.csv")
if isfile(result_path)
    error("file $result_path already exists")
end

perf_list = []

for train_method in [:Handwritten, :Super_Hand, :Super_TV, :Super_noiseless, :EM] # :VI
    # you can find the available args inside `train_models.jl`.
    global script_args = (;
        is_quick_test=false,
        scenario=SEDL.HovercraftScenario(),
        gpu_id=5,
        # load_trained=true, 
        use_fixed_variance=false,
        use_simple_obs_model=false,
        train_method,
    )
    my_include("../train_models.jl")
    push!(perf_list, Main.perf)
end

results = DataFrame(perf_list)
results|> display
CSV.write(result_path, results)