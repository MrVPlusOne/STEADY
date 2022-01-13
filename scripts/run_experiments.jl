
my_include = include # to avoid mess up the VSCode linter

perf_list = []
for train_method in [:VI] # [:Supervised, :EM, :VI]
    # you can find the available args inside `train_models.jl`.
    global script_args = (;
        load_trained=true,
        is_quick_test=true,
        gpu_id=0,
        use_sim_data=false,
        use_simple_obs_model=true,
        train_method,
    )
    my_include("train_models.jl")
    push!(perf_list, perf)
end

DataFrame(perf_list) |> display