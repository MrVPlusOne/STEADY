using DataFrames
using CSV
include("experiment_common.jl")

training_curves = []
particle_sizes = [200, 2_000, 20_000, 200_000]
exp_group = "vary_particle_size"
result_dir = joinpath("reports/$exp_group")
mkpath(result_dir)

with_alert("particle_variation.jl") do
    for n_particles in particle_sizes
        # you can find the available args inside `train_models.jl`.
        local script_args = (;
            # is_quick_test=true,
            scenario=SEDL.RealCarScenario("alpha_truck"),
            n_particles,
            train_method=:EM,
            gpu_id=Main.GPU_ID,
            max_train_steps=120_000,
            exp_group,
        )
        @eval(Main, script_args = $script_args)
        dynamic_include("../train_models.jl")
        push!(training_curves, Main.training_curve)
        label="n_particle=$(n_particles/1000)K"
        CSV.write(joinpath(result_dir, "$label.csv"), DataFrame(Main.training_curve))
    end
end