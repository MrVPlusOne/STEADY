##-----------------------------------------------------------
# imports
using StatsPlots

StatsPlots.default(dpi = 300, legend = :outerbottom)
##-----------------------------------------------------------
# generate data
# state vec represents pos and vel.
quick_test = false

x0_dist = MvNormal([1.0, 0.0], 0.4)

function motion_model(x::X, control, Δt) where X
    μ = x + X([x[2], control[1] / 2]) * Δt
    MvNormal(μ, X([0.1, 0.5]) * Δt)
end

function obs_model(x)
    Normal(x[1], 0.1)
end

function controller(x, obs, t)
    [(-1 - x[2]) + randn() * 0.3]
end

system = MarkovSystem(x0_dist, motion_model, obs_model)
Δt = 0.1
times = 0:Δt:8 |> collect
Δt = Float32(Δt)

sim_data = simulate_trajectory(times, [1.0, 0.0], system, controller)

plot(times, hcatreduce(sim_data.states)', label = ["x (truth)" "v (truth)"]) |> display
##-----------------------------------------------------------
function plot_trajectories!(
    comp_names::AbsVec{String}, trajectories::AbsVec{<:AbsVec};
)
    linealpha = 1.0 / sqrt(length(trajectories))
    concat_traj = typeof(trajectories[1][1])[]
    concat_times = Float64[]
    for tr in trajectories
        append!(concat_traj, tr)
        append!(concat_times, times)
        push!(concat_traj, fill(NaN, length(tr[1])))
        push!(concat_times, NaN)
    end
    ys = hcatreduce(concat_traj)'
    plot!(concat_times, ys; label = permutedims(comp_names), linealpha)
end

obs_frames = [1, 50]
obs_data = (; times, sim_data.observations, sim_data.controls, obs_frames, x0_dist)

post_sampler = ParticleFilterSampler(
    n_particles = quick_test ? 2000 : 50_000,
    n_trajs = 100,
)

sampling_result = sample_posterior(
    post_sampler, system, obs_data, new_state(post_sampler); showprogress = true)
plot(times, hcatreduce(sim_data.states)', label = ["x (truth)" "v (truth)"])
plot_trajectories!(["x (post)", "v (post)"], sampling_result.trajectories) |> display
##-----------------------------------------------------------
using CUDA
use_gpu = false

x_dim = 2
rnn_dim = 64
observation_vec = map(1:length(times)) do t
    u = obs_data.controls[t]
    has_obs = t in obs_data.obs_frames
    y = obs_data.observations[t]
    y = has_obs ? y : zero(y)
    [has_obs; u; y]
end
y_dim = length(observation_vec[1])

guide = mk_guide(x_dim, y_dim, rnn_dim) |> to_device(use_gpu)
vi_smoother = VISmoother(; guide, on_gpu = use_gpu)

init_trajs = [vi_smoother(observation_vec, Δt).trajectory for i in 1:100]
plot(times, hcatreduce(sim_data.states)', label = ["x (truth)" "v (truth)"])
plot_trajectories!(["x (init)", "v (init)"], init_trajs) |> display
##-----------------------------------------------------------
# train the guide
log_joint = let odata = obs_data
    (traj, observations) -> begin
        states = traj
        states_logp(motion_model, odata, states) + data_logp((;obs_model), odata, states)
    end
end

elbo_history = []
adam=ADAM(1e-4)
let n_steps=200, prog = Progress(n_steps)
    @time train_guide!(vi_smoother, log_joint, observation_vec, Δt; 
        optimizer=adam,
        n_steps,
        callback=r -> begin
            push!(elbo_history, r.elbo)
            if r.step % 50 == 1
                sampled_trajs = [vi_smoother(observation_vec, Δt).trajectory for i in 1:100]
                plot(times, hcatreduce(sim_data.states)', 
                    label = ["x (truth)" "v (truth)"], title="iteration $(r.step)")
                plot_trajectories!(["x (trained)", "v (trained)"], sampled_trajs) |> display
            end
            next!(prog, showvalues = [(:elbo, r.elbo)])
        end,
        n_samples = 16,
    )
end

plot(elbo_history, label="ELBO") |> display
##-----------------------------------------------------------
# plot trained posterior trajectories
g1 = ones(10) |> Flux.gpu
randn!(zero(g1))
##-----------------------------------------------------------