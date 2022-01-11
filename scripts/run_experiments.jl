
my_include = include # to avoid mess up with VSCode linter

perf_list = []
for train_method in [:Supervised, :EM, :VI]
    script_args = (;
        is_quick_test=false,
        gpu_id=0,
        use_sim_data=false,
        use_simple_obs_model=false,
        train_method,
    )
    my_include("train_models.jl")
    push!(perf_list, perf)
end

DataFrame(perf_list) |> display