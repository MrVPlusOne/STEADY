include("experiment_common.jl")

training_curves = []
particle_sizes = [2_000, 20_000, 200_000]

for n_particles in particle_sizes
    # you can find the available args inside `train_models.jl`.
    local script_args = (;
        # is_quick_test=true,
        scenario=SEDL.RealCarScenario("ut_automata"),
        n_particles,
        exp_name="particles=$(n_particles/1000)K",
        train_method=:EM,
        gpu_id=Main.GPU_ID,
        max_train_steps=120_000,
    )
    @eval(Main, script_args = $script_args)
    dynamic_include("../train_models.jl")
    push!(training_curves, Main.training_curve)
end

using DataFrames
using CSV

let result_dir = joinpath("results/vary_particle_size")
    mkpath(result_dir)
    step_plt = plot()
    time_plt = plot()
    for (n_particle, curve) in zip(particle_sizes, training_curves)
        @unzip_named (xs, :step), (times, :training_time), (ys, :RMSE) = curve
        label="n_particle=$(n_particle/1000)K"
        times = times .- times[1] # remove the initialization time.
        half_steps = length(xs) ÷ 2
        plot!(step_plt, xs[half_steps:end], ys[half_steps:end]; xlabel="step", ylabel="RMSE", label)
        plot!(time_plt, times, ys; xlabel="training time (s)", ylabel="RMSE", label)
        CSV.write(joinpath(result_dir, "$label.csv"), DataFrame(curve))
    end
    savefig(step_plt, joinpath(result_dir, "perf_vs_step.png"))
    savefig(time_plt, joinpath(result_dir, "perf_vs_time.png"))
end

alert("particle_variation.jl finished.")