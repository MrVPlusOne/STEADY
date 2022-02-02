using DataFrames, CSV

my_include = include # to avoid mess up the VSCode linter

perf_list = []

let ° = π / 180

    for σ_bearing in [1°] # [5°, 1°, 2.5°, 10°, 20°]
        # you can find the available args inside `train_models.jl`.
        global script_args = (;
            gpu_id=7,
            σ_bearing,
            exp_name="obs_weight=0.1, σ_bearing=$(σ_bearing/°)°",
        )
        my_include("../train_models.jl")
        push!(perf_list, Main.perf)

        dest_dir = SEDL.data_dir("obs_noise_variation/$(Main.exp_name)")
        isdir(dest_dir) && rm(dest_dir; recursive=true)
        mkpath(dirname(dest_dir))
        run(`cp -r "$(Main.save_dir)/tb_logs" "$dest_dir"`)
    end
end

result_path = joinpath("results", "obs_noise_variation.csv")
DataFrame(perf_list) |> display
CSV.write(result_path, DataFrame(perf_list))