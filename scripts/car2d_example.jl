##-----------------------------------------------------------
using Distributions
using StatsPlots
import Random

StatsPlots.default(dpi=600, legend=:outerbottom)
##-----------------------------------------------------------

##-----------------------------------------------------------
# generate data
Random.seed!(123)
times = collect(0.0:0.1:10.0)
true_params = (; τ_v = 0.8, l = 0.5, fraction_vy=0.5, inertia_vy=0.2)
true_σ = (; σ_pos=@SVector[0.05, 0.05], σ_θ=0.1, σ_v=@SVector[0.2, 0.1])
tiny_σ = (; σ_pos=@SVector[0.01, 0.01], σ_θ=0.01, σ_v=@SVector[0.01, 0.01])
true_motion_model = Car2D.car2d_motion_model(
    Car2D.car_velocity_f(true_params), true_σ)

vdata = Car2D.variable_data()
x0_dist = init_state_distribution(vdata)
params_dist = params_distribution(vdata)

landmarks = @SVector[@SVector[1.0, 2.5], @SVector[6.0, 1.5], @SVector[4.0, -2.0]]
noise_scale = 1.0
true_system = MarkovSystem(x0_dist,true_motion_model, 
    Car2D.sensor_dist(landmarks, noise_scale))

ex_data = simulate_trajectory(times, true_system, Car2D.manual_control())
obs_data = (; times, ex_data.observations, ex_data.controls)
@df ex_data.controls plot(times, [:v̂, :ϕ̂], label=["v̂" "ϕ̂"])
let traj_p = plot()
    Car2D.plot_states!(ex_data.states, "truth")
    @unzip xs, ys = landmarks
    scatter!(xs, ys, label="Landmarks")
    plot(traj_p, legend=:outerbottom, aspect_ratio=1.0)
end |> display
##-----------------------------------------------------------
ffbs_result = @time ffbs_smoother(true_system, obs_data, n_particles=10_000, n_trajs=100)
let plt = plot()
    Car2D.plot_particles!(ffbs_result.particles, "FFBS", ex_data.states)
    @unzip xs, ys = landmarks
    scatter!(xs, ys, label="Landmarks")
    display("image/png", plt)
end
##-----------------------------------------------------------