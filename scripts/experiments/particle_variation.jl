include("experiment_common.jl")

training_curves = []
particle_sizes = [200, 2_000, 20_000, 200_000]
exp_group = "vary_particle_size"

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
    end
end

# plot data
using DataFrames
using CSV

let result_dir = joinpath("reports/$exp_group")
    mkpath(result_dir)
    step_plt = plot(legend=:topright)
    time_plt = plot(legend=:topright)
    for (n_particle, curve) in zip(particle_sizes, training_curves)
        @unzip_named (xs, :step), (times, :training_time), (ys, :total) = curve
        label="n_particle=$(n_particle/1000)K"
        times = times .- times[1] # remove the initialization time.
        ids = filter(i -> 3_000 <= xs[i] <= 30_000, eachindex(xs))
        tids = filter(i -> 400 <= times[i] <= 5000, eachindex(xs))
        plot!(step_plt, xs[ids], ys[ids]; xlabel="step", ylabel="State Estimation Error", label)
        plot!(time_plt, times[tids], ys[tids]; xlabel="training time (s)", ylabel="State Estimation Error", label)
        CSV.write(joinpath(result_dir, "$label.csv"), DataFrame(curve))
    end
    savefig(step_plt, joinpath(result_dir, "perf_vs_step.pdf"))
    savefig(time_plt, joinpath(result_dir, "perf_vs_time.pdf"))
end

alert("particle_variation.jl finished.")