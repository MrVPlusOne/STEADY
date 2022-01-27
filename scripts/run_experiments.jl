my_include = include # to avoid mess up the VSCode linter

perf_list = []
for train_method in [:Super_noiseless] #[:Super_noiseless, :Handwritten, :Super_Hand, :Super_TV, :EM, :VI]
    # you can find the available args inside `train_models.jl`.
    global script_args = (;
        is_quick_test=false,
        gpu_id=1,
        # load_trained=true, 
        use_sim_data=false,
        use_simple_obs_model=false,
        train_method,
    )
    my_include("train_models.jl")
    push!(perf_list, Main.perf)
end

DataFrame(perf_list) |> display