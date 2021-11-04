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
    n_threads=min(Threads.nthreads()-1, length(systems)),
)
    n_runs = length(systems)
    @assert n_runs == length(systems) == length(obs_data_list)

    results = parallel_map(1:n_runs, nothing; n_threads) do _, i
        sample_posterior(sampler, systems[i], obs_data_list[i], states[i], 
            showprogress=i==1)
    end
    @unzip_named (log_obs, :log_obs), (trajectories_list, :trajectories) = results
    trajectories = reduce(hcat, trajectories_list)
    (;trajectories, log_obs)
end