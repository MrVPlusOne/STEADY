abstract type PosteriorSampler end

"""
    new_state(::PosteriorSampler) -> sampler_state
"""
function new_state end

"""
    sample_posterior(sampler, sampler_state, system, obs_data, showprogress) -> (; trajectories, log_obs)
where `trajectories::Vector{Vector{State}}` and `log_obs::Vector{Float64}`.
"""
function sample_posterior end

@kwdef(
struct ParticleFilterSampler <: PosteriorSampler
    n_particles::Int
    n_trajs::Int
end)

new_state(::ParticleFilterSampler) = Dict{Symbol, Any}()

function sample_posterior(
    pf::ParticleFilterSampler, system, obs_data, state::Dict; showprogress,
)
    (; n_particles, n_trajs) = pf
    filter_smoother(system, obs_data, use_auxiliary_proposal=false;
        n_particles, n_trajs, cache=state, showprogress)
end

function sample_posterior_parallel(
    sampler::PosteriorSampler, systems::Vector, obs_data_list::Vector,
    states::Vector=[new_state(sampler) for _ in systems];
    n_threads=min(Threads.nthreads()-1, length(systems), 10),
)
    n_runs = length(systems)
    @assert n_runs == length(systems) == length(obs_data_list)

    progress = Progress(length(systems), desc="sample_posterior_parallel")
    results = parallel_map(1:n_runs, nothing; n_threads) do _, i
        r = sample_posterior(sampler, systems[i], obs_data_list[i], states[i], 
            showprogress=false)
        next!(progress)
        r
    end
    @unzip_named (log_obs, :log_obs), (trajectories_list, :trajectories) = results
    trajectories = reduce(hcat, trajectories_list)
    n_effective = sum(x -> x.n_effective, results)
    (;trajectories, n_effective, log_obs)
end

@kwdef(
struct ParticleGibbsSampler <: PosteriorSampler
    n_particles::Int
    n_jumps::Int
    thining::Int
    resample_threshold::Float64=0.5
end)

new_state(::ParticleGibbsSampler) = Dict{Symbol, Any}()

function sample_posterior(
    pf::ParticleGibbsSampler, system, obs_data, state::Dict; showprogress,
)
    if isempty(state)
        showprogress && @info "Sampling initial trajectory using paritcle filter..."
        (; trajectories) = filter_smoother(system, obs_data, use_auxiliary_proposal=false;
            pf.n_particles, n_trajs=1, pf.resample_threshold, showprogress)
        state[:trajs] = [trajectories[1]]
    end

    kernel = PGAS_kernel(system, obs_data; 
        showprogress=false, pf.n_particles, pf.resample_threshold)

    trajectories = state[:trajs]
    new_traj = trajectories[end]

    showprogress && (progress = Progress(pf.n_jumps, desc="sample_posterior", output=stdout))
    for i in 1:pf.n_jumps
        new_traj = kernel(new_traj)
        if i % pf.thining == 0
            push!(trajectories, new_traj)
        end
        showprogress && next!(progress)
    end
    (; trajectories, log_obs=0.0)
end