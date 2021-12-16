##-----------------------------------------------------------
# imports
using StatsPlots

StatsPlots.default(; dpi=300, legend=:outerbottom)
##-----------------------------------------------------------
# generate data
# state vec represents pos and vel.
quick_test = false

x0_dist = SMvNormal([1.0, 0.0], 0.4)

function motion_model(x::X, control, Δt) where {X}
    μ = x + X([x[2], control[1] / 2]) * Δt
    MvNormal(μ, X([0.1, 0.5]) * Δt)
end


function traj_logp(obs_data, Δt, device)
    (; controls, observations, obs_frames) = obs_data
    x0_μ = [1.0, 0.0] |> device
    x0_σ = [0.4, 0.4] |> device
    traj_encs -> begin
        lp = logpdf_normal(x0_μ, x0_σ, traj_encs[1])
        batch_size = length(lp)
        for t in 2:length(traj_encs)
            uvec = repeat(device(controls[t - 1]), 1, batch_size)
            μ = traj_encs[t - 1] + vcat(traj_encs[t - 1][2:2, :], uvec / 2) * Δt
            lp += logpdf_normal(μ, device([0.1, 0.5]) * Δt, traj_encs[t])
        end
        for t in 1:length(traj_encs)
            if t in obs_frames
                lp += logpdf_normal(traj_encs[t][1:1, :], 0.1, device(observations[t]))
            end
        end
        lp
    end
end

function obs_model(x)
    Normal(x[1], 0.1)
end

function controller(x, obs, t)
    [(-1 - x[1]) + randn() * 0.3]
end

system = MarkovSystem(x0_dist, motion_model, obs_model)
Δt = 0.1
times = 0:Δt:8 |> collect
Δt = Float32(Δt)

sim_data = simulate_trajectory(times, [1.0, 0.0], system, controller)

plot(times, hcatreduce(sim_data.states)'; label=["x (truth)" "v (truth)"]) |> display
plot(times, hcatreduce(sim_data.controls)'; label=["control"]) |> display
plot(times, hcatreduce(sim_data.observations)'; label=["obs"]) |> display
##-----------------------------------------------------------
function plot_trajectories!(comp_names::AbsVec{String}, trajectories::AbsVec{<:AbsVec};)
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
    plot!(concat_times, ys; label=permutedims(comp_names), linealpha, linecolor=[1 2])
end

obs_frames = [1:4:26..., 60:4:80...]
obs_data = (; times, sim_data.observations, sim_data.controls, obs_frames, x0_dist)

post_sampler = ParticleFilterSampler(; n_particles=quick_test ? 2000 : 50_000, n_trajs=100)

sampling_result = sample_posterior(
    post_sampler, system, obs_data, new_state(post_sampler); showprogress=true
)
plot(times, hcatreduce(sim_data.states)'; label=["x (truth)" "v (truth)"])
plot_trajectories!(["x (post)", "v (post)"], sampling_result.trajectories) |> display
##-----------------------------------------------------------
using CUDA
use_gpu = true

function mk_observation_vec(obs_data)
    map(1:length(obs_data.times)) do t
        has_obs = t in obs_data.obs_frames
        if has_obs
            u = obs_data.controls[t]
            y = obs_data.observations[t]
            Some([u; y])
        else
            nothing
        end
    end
end
observation_vec = mk_observation_vec(obs_data)

x_dim = 2
h_dim = 64
y_dim = length(obs_data.observations[1]) + length(obs_data.controls[1])

guide = mk_guide(x_dim, y_dim, h_dim) |> to_device(use_gpu)
vi_smoother = VISmoother(; guide, on_gpu=use_gpu)

init_trajs = vi_smoother(observation_vec, Δt, 100).trajectories
plot(times, hcatreduce(sim_data.states)'; label=["x (truth)" "v (truth)"])
plot_trajectories!(["x (init)", "v (init)"], init_trajs) |> display

elbo_history = []
adam = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.WeightDecay(1e-4), ADAM(1e-4))
##-----------------------------------------------------------
# train the guide
log_joint = traj_logp(obs_data, Δt, to_device(use_gpu))

linear(from, to) = x -> from + (to - from) * x

let n_steps = 2001, prog = Progress(n_steps)
    @info "Training the guide..."
    train_result = @time train_guide!(
        vi_smoother,
        log_joint,
        observation_vec,
        Δt;
        optimizer=adam,
        n_steps,
        anneal_schedule=step -> linear(1e-3, 1.0)(min(1, 2step / n_steps)),
        callback=r -> begin
            push!(elbo_history, r.elbo)
            if r.step % 50 == 1
                sampled_trajs = vi_smoother(observation_vec, Δt, 100).trajectories
                plot(
                    times,
                    hcatreduce(sim_data.states)';
                    label=["x (truth)" "v (truth)"],
                    title="iteration $(r.step)",
                )
                plot_trajectories!(["x (trained)", "v (trained)"], sampled_trajs) |>
                display
            end
            next!(
                prog;
                showvalues=[(:elbo, r.elbo), (:step, r.step), (:batch_size, r.batch_size)],
            )
        end,
        n_samples_f=step -> ceil(Int, linear(64, 512)(step / n_steps)),
        # n_samples_f = step -> 512,
    )
    display(train_result)
end

plot(elbo_history; label="ELBO") |> display
##-----------------------------------------------------------
test_data = simulate_trajectory(times, [0.9, 1.1], system, controller)

let
    test_obs_data = (;
        times, test_data.observations, test_data.controls, obs_frames, x0_dist
    )
    test_trajs = vi_smoother(mk_observation_vec(test_obs_data), Δt, 100).trajectories
    plot(
        times,
        hcatreduce(test_data.states)';
        label=["x (truth)" "v (truth)"],
        title="test result",
    )
    plot_trajectories!(["x (trained)", "v (trained)"], test_trajs) |> display
end
##-----------------------------------------------------------