my_include = include # to avoid mess up the VSCode linter

perf_list = []
for n_particles in [1000, 10_000, 100_000] #[:Super_noiseless, :Handwritten, :Super_Hand, :Super_TV, :EM, :VI]
    # you can find the available args inside `train_models.jl`.
    global script_args = (;
        is_quick_test=false,
        gpu_id=1,
        use_simple_obs_model=false,
        n_particles,
        exp_name="particles=$n_particles",
    )
    my_include("../train_models.jl")
    push!(perf_list, Main.perf)

    dest_dir = SEDL.data_dir("particle_ablation/$(Main.exp_name)")
    isdir(dest_dir) && rm(dest_dir, recursive=true)
    mkpath(dest_dir)
    run(`cp -r "$(Main.save_dir)/tb_logs" "$dest_dir"`)
end

DataFrame(perf_list) |> display