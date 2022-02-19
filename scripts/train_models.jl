##-----------------------------------------------------------
# imports
using DrWatson
using Revise
using Flux: Flux
using CUDA: CUDA, CuArray
CUDA.allowscalar(false)
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)
using TensorBoardLogger: TBLogger
using ProgressMeter
using Serialization: serialize, deserialize
using DataFrames: DataFrame
using Alert
using Random
using Statistics
using Setfield
using Distributions: logpdf
using CSV: CSV
using LinearAlgebra

include("experiments/experiment_common.jl")
include("data_from_source.jl")
##-----------------------------------------------------------
# set up scenario and simulate some data

if !isdefined(Main, :script_args)
    # script_args can be used to override the default config parameters.
    script_args = (
        is_quick_test=true,
        gpu_id=0,
        scenario=SEDL.RealCarScenario("ut_automata"),
        # use_simple_obs_model=true,
        train_method=:EM_SLAM,
        n_particles=1000,
    )
end

(;
    scenario,
    is_quick_test,
    load_trained,
    should_train_dynamics,
    gpu_id,
    use_simple_obs_model,
    σ_bearing,
    use_fixed_variance,
    train_method,
    n_train_ex,
    validation_metric,
    lr,
    max_obs_weight,
    use_obs_weight_schedule,
    max_train_steps,
    exp_name,
    n_particles,
    h_dim,
    run_id,
) = let
    config = (;
        scenario=SEDL.RealCarScenario("alpha_truck"),
        is_quick_test=false,
        load_trained=false,
        should_train_dynamics=true, # whether to train a NN motion model or use the ground truth
        gpu_id=nothing, # integer or nothing
        use_simple_obs_model=false,
        σ_bearing=5°,
        use_fixed_variance=false,
        train_method=:EM, # see `AllTrainingMethods`.
        n_train_ex=16,  # number of training trajectories when using simulation data
        validation_metric=:RMSE,  # :RMSE or :log_obs
        lr=1e-4,
        max_obs_weight=1.0,
        use_obs_weight_schedule=true, # whether to increase obs_weight from 0 to max_obs_weight over time
        max_train_steps=40_000,
        exp_name=nothing,
        n_particles=20_000,  # how many particles to use for the EM training.
        h_dim=64,
        run_id=1,
    )
    # discard params that are the same as the default
    global script_args = let changes = []
        foreach(keys(script_args)) do k
            @smart_assert k in keys(config)
            if script_args[k] != config[k]
                push!(changes, k => script_args[k])
            end
        end
        (; changes...)
    end
    merge(config, script_args::NamedTuple)
end
check_training_method(train_method)
is_quick_test && @info "Quick testing model training..."

println("--------------------")
println("training settings: ")
for (k, v) in pairs(script_args)
    println("  $k: $v")
end
println("--------------------")

Random.seed!(123 + run_id)

use_gpu = gpu_id !== nothing
if use_gpu
    CUDA.device!(gpu_id)
    @info "Using GPU #$(gpu_id)"
    CUDA.seed!(456 + run_id)
else
    @warn "gpu_id = $gpu_id, using CPU."
end
device = use_gpu ? Flux.gpu : (Flux.f32 ∘ Flux.cpu)
tconf = SEDL.TensorConfig(use_gpu, Float32)

use_sim_data = !(scenario isa SEDL.RealCarScenario)
data_source = if use_sim_data
    SimulationData(;
        n_train_ex, n_valid_ex=min(n_train_ex, 32), n_test_ex=32, times=0:tconf(0.1):10
    )
else
    @smart_assert scenario isa SEDL.RealCarScenario
    if scenario.data_name == "alpha_truck"
        # lagacy format
        SeparateData(;
            train_data_dir="datasets/alpha_truck/train",
            valid_data_dir="datasets/alpha_truck/valid",
            test_data_dir="datasets/alpha_truck/test",
        )
    else
        MixedData(;
            data_dir="datasets/$(scenario.data_name)",
            test_data_ratio=0.25,
            valid_data_ratio=0.25,
        )
    end
end

state_L2_loss = SEDL.state_L2_loss_batched(scenario)

sketch = SEDL.batched_sketch(scenario)

# temporarily hardcoded
landmarks = if use_sim_data
    [[12.0, -8.0], [-6.0, -4.0], [-4.0, 10.0], [6.0, 7.0]]
else
    if scenario.data_name == "alpha_truck"
        # lagacy format
        [[-1.230314, -0.606814], [0.797073, 0.889471], [-3.496525, 0.207874], [0.0, 6.0]]
    elseif scenario.data_name == "ut_automata"
        [[1.2, -1], [0.6, -4], [-1.5, 1.1], [-4.2, -1.0]]
    else
        error("Not implemented.")
    end
end

if use_simple_obs_model
    obs_model = let σs = (pos=0.2, angle_2d=0.2, vel=0.2, ω=0.2)
        state -> SEDL.gaussian_obs_model(state, σs)
    end
else
    landmarks_to_obs_model =
        landmarks -> x -> SEDL.landmark_obs_model_warp(x, (; landmarks, σ_bearing))
    obs_model = landmarks_to_obs_model(SEDL.landmarks_to_tensor(tconf, landmarks))
end
logpdf_obs(x, y) = logpdf(obs_model(x), y)
landmarks_to_logpdf_obs = landmarks -> begin
    obs_m = landmarks_to_obs_model(landmarks)
    (x, y) -> logpdf(obs_m(x), y)
end

data_cache_path = let
    cache_name = savename(
        (;
            scenario=summary(scenario),
            source=string(data_source),
            use_simple_obs_model,
            σ_bearing,
            data_source,
        ),
        "serial";
        connector="-",
    )
    SEDL.data_dir("data_cache", cache_name)
end
datasets = if data_source isa SimulationData
    true_params = map(p -> convert(tconf.ftype, p), SEDL.simulation_params(scenario))
    motion_model = SEDL.BatchedMotionModel(
        tconf, sketch, SEDL.batched_core(scenario, true_params)
    )
    # cache results to ensure reproducibility
    data_cpu = generate_or_load(data_cache_path, "datasets") do
        data_from_source(scenario, data_source, tconf; motion_model, obs_model)
    end
else
    data_cpu = generate_or_load(data_cache_path, "datasets") do
        data_from_source(scenario, data_source, tconf; obs_model)
    end
end

data_train, data_valid, data_test = datasets.train, datasets.valid, datasets.test

@smart_assert data_train.states[1].tconf == tconf
n_train_ex = data_train.states[1].batch_size
obs_frames = eachindex(data_train.times)

let
    plot()
    SEDL.plot_2d_landmarks!(landmarks, "truth"; color=2)
    SEDL.plot_2d_trajectories!(data_train.states, "truth"; linecolor=1)
end |> display

SEDL.plot_batched_series(
    data_train.times, getindex.(data_train.states, 1); title="States"
) |> display
SEDL.plot_batched_series(
    data_train.times, getindex.(data_train.controls, 1); title="Controls"
) |> display
SEDL.plot_batched_series(
    data_train.times, getindex.(data_train.observations, 1); title="Observations"
) |> display
##-----------------------------------------------------------
# utilities
function copy_model(model)
    if use_gpu
        Flux.cpu(model)
    else
        deepcopy(model)
    end
end

"""
Plot the posterior trajectories sampled by a particle filter.
"""
function plot_posterior(
    motion_model,
    data,
    sample_id=1;
    n_particles=100_000,
    n_trajs=100,
    obs_frames=nothing,
    plot_args...,
)
    pf_trajs = SEDL.sample_posterior_pf(
        motion_model, logpdf_obs, data, sample_id; n_particles, n_trajs, obs_frames
    )
    SEDL.plot_batched_series(
        data.times, pf_trajs; truth=getindex.(data.states, sample_id), plot_args...
    )
end

function plot_core_io(
    motion_model::BatchedMotionModel,
    data,
    sample_id=1;
    n_particles=100_000,
    n_trajs=100,
    obs_frames=nothing,
    plot_args...,
)
    pf_trajs, core_in, core_out = SEDL.sample_posterior_pf(
        motion_model,
        logpdf_obs,
        data,
        sample_id;
        n_particles,
        n_trajs,
        obs_frames,
        record_io=true,
    )
    series = map(merge, core_in, core_out)
    core_in_truth, core_out_truth = SEDL.input_output_from_trajectory(
        motion_model.sketch,
        getindex.(data.states, sample_id),
        getindex.(data.controls, sample_id),
        data.times,
    )
    truth = map(merge, core_in_truth, core_out_truth)
    plot_batched_series(data.times[1:(end - 1)], series; truth, plot_args...)
end

function posterior_metrics(
    motion_model, data; n_repeats=1, obs_frames=1:10:length(data.times), n_particles=100_000
)
    rows = @showprogress 0.1 "posterior_metrics" map(1:n_repeats) do _
        SEDL.estimate_posterior_quality(
            motion_model, logpdf_obs, data; n_particles, obs_frames, state_L2_loss
        )
    end
    if n_repeats == 1
        rows[1]
    else
        SEDL.named_tuple_reduce(rows, mean)
    end
end

function with_alert(task::Function, task_name::String)
    try
        local result = task()
        is_quick_test || alert("$task_name finished. Setting: $script_args.")
        result
    catch e
        if e isa InterruptException
            throw(InterruptException()) # don't need the notification and stack trace.
        end
        alert("$task_name stopped due to exception: $(summary(e)).")
        rethrow()
    end
end

function save_model_weights!()
    mm_weights = Flux.params(Main.learned_motion_model)
    serialize(joinpath(save_dir, "mm_weights.serial"), mm_weights)
    if train_method == :SVI
        guide_weights = Flux.params(guide)
        serialize(joinpath(save_dir, "guide_weights.serial"), guide_weights)
    end
    if train_method == :EM_SLAM
        serialize(joinpath(save_dir, "landmark_guess.serial"), landmark_guess)
        serialize(joinpath(save_dir, "x0_dists.serial"), Flux.params(x0_dists))
    end
end

function load_model_weights!()
    mm_weights = deserialize(joinpath(save_dir, "mm_weights.serial"))
    Flux.loadparams!(Main.learned_motion_model, device(mm_weights))
    if train_method == :SVI
        guide_weights = deserialize(joinpath(save_dir, "guide_weights.serial"))
        Flux.loadparams!(guide, device(guide_weights))
    end
    if train_method == :EM_SLAM
        Main.landmark_guess .= deserialize(joinpath(save_dir, "landmark_guess.serial"))
        Flux.loadparams!(
            x0_dists, device(deserialize(joinpath(save_dir, "x0_dists.serial")))
        )
    end
end

function save_best_model(
    es::SEDL.EarlyStopping, metrics::NamedTuple, model_info::NamedTuple
)
    loss = if validation_metric == :RMSE
        metrics.RMSE
    elseif validation_metric == :log_obs
        -metrics.log_obs
    else
        error("Not implemented for metric: $validation_metric.")
    end

    es(loss, model_info, save_model_weights!)
end

use_sim_data && display(plot_posterior(motion_model, data_train; title="true posterior"))
##-----------------------------------------------------------
# load or set up the NN models

y_dim = sum(m -> size(m, 1), data_train.observations[1].val)

save_dir = let prefix = is_quick_test ? "sims-quick" : "sims", postfix = "run-$(run_id)"
    save_args = SEDL.dropnames(script_args, (:gpu_id, :is_quick_test, :run_id))
    SEDL.data_dir(
        prefix,
        savename("train_models-$(summary(scenario))", save_args; connector="-"),
        postfix,
    )
end

if !load_trained && isdir(save_dir)
    @warn "removing old data at '$save_dir' ..."
    rm(save_dir; recursive=true)
end
mkpath(save_dir)
write(joinpath(save_dir, "settings.txt"), string(script_args))

logger = TBLogger(joinpath(save_dir, "tb_logs"))
println("""To view tensorboard logs, use the following command:
```
tensorboard --samples_per_plugin "images=100" --logdir "$save_dir/tb_logs"
```""")

@info "Computing normal transforms..."
normal_transforms = @time SEDL.compute_normal_transforms(
    sketch, data_train.states, data_train.controls, data_train.observations, data_train.Δt
)

learned_motion_model = if should_train_dynamics
    nn_motion_model =
        SEDL.mk_nn_motion_model(;
            sketch, tconf, h_dim, use_fixed_variance, normal_transforms
        ) |> device
    @smart_assert !isempty(Flux.params(nn_motion_model))
    nn_motion_model
else
    motion_model
end

plot_posterior(
    learned_motion_model, data_train; obs_frames=1:1, title="Motion model prior (initial)"
) |> display

if train_method == :SVI
    guide = SEDL.mk_guide(; sketch, h_dim, y_dim, normal_transforms) |> device

    SEDL.plot_guide_posterior(guide, data_train, 1; title="Guide posterior (initial)") |>
    display
end

if train_method == :EM_SLAM
    landmark_guess = SEDL.landmarks_to_tensor(tconf, [randn(size(l)) for l in landmarks])

    x0_dists = map(1:n_train_ex) do _
        Σ = Diagonal(5 * ones(6, 6))
        SEDL.SE2Distribution(tconf, randn(6), Σ)
    end
end

if load_trained && train_method != :Handwritten
    @warn "Loading motion model weights from file..."
    load_model_weights!()
end

adam = Flux.ADAM(lr)
##-----------------------------------------------------------
# train the model using expectation maximization

function em_callback(
    learned_motion_model::BatchedMotionModel,
    early_stopping::SEDL.EarlyStopping;
    n_steps,
    test_every=100,
    landmark_est=nothing,
)
    prog = Progress(n_steps; showspeed=true)
    function (r)
        Base.with_logger(logger) do
            @info "training" r.log_obs r.loss r.lr r.obs_weight
        end

        # Compute test log_obs and plot a few trajectories.
        if r.step % test_every == 1 || r.step == n_steps
            valid_metrics = posterior_metrics(learned_motion_model, data_valid)
            to_stop = save_best_model(early_stopping, valid_metrics, (step=r.step,))

            Base.with_logger(logger) do
                @info "validation" valid_metrics... log_step_increment = 0
            end

            # plot 2d trajectories
            scenario_plt = plot()
            SEDL.plot_2d_landmarks!(landmarks, "truth"; color=2)
            if landmark_est !== nothing
                SEDL.plot_2d_landmarks!(
                    SEDL.landmarks_from_tensor(landmark_est),
                    "estimated";
                    color=4,
                    marker=:auto,
                )
                logpdf_obs_est = landmarks_to_logpdf_obs(landmark_est)
            else
                logpdf_obs_est = logpdf_obs
            end

            Base.with_logger(logger) do
                @info "training" landmarks=scenario_plt log_step_increment = 0
            end

            let pf_trajs = SEDL.sample_posterior_pf(
                    learned_motion_model, logpdf_obs_est, data_train, 1; obs_frames
                )
                SEDL.plot_2d_trajectories!(pf_trajs, "posterior"; linecolor=1)
            end
            SEDL.plot_2d_trajectories!(getindex.(data_train.states, 1), "truth"; linecolor=3)


            Base.with_logger(logger) do
                @info "training" scenario_plt log_step_increment = 0
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? data_train : data_test
                plt = plot_posterior(
                    learned_motion_model,
                    sim_data,
                    id;
                    title="$name traj $id (iter $(r.step))",
                )

                Base.with_logger(logger) do
                    kv = [Symbol("traj_$id") => plt]
                    @info name log_step_increment = 0 kv...
                end
                # display(plt)
            end
        else
            to_stop = (; should_stop=false)
        end

        next!(
            prog;
            showvalues=[
                (:log_obs, r.log_obs),
                (:loss, r.loss),
                (:step, r.step),
                (:learning_rate, r.lr),
            ],
        )
        to_stop
    end
end

if !load_trained && (train_method == :EM)
    with_alert("EM training") do
        n_steps = is_quick_test ? 3 : max_train_steps + 1
        es = SEDL.EarlyStopping(; patience=40)
        obs_weight_schedule = if use_obs_weight_schedule
            step -> linear(1e-3, 1.0)(min(1.0, step / n_steps)) * max_obs_weight
        else
            step -> max_obs_weight
        end
        @info "Training the dynamics using EM"
        SEDL.train_dynamics_EM!(
            learned_motion_model,
            logpdf_obs,
            data_train.states[1],
            data_train.observations,
            data_train.controls,
            (; data_train.times, obs_frames);
            optimizer=adam,
            n_steps,
            n_particles,
            callback=em_callback(learned_motion_model, es; n_steps, test_every=500),
            obs_weight_schedule,
        )
        @info "Best model: $(es.model_info)"
    end
    load_model_weights!()
end
##-----------------------------------------------------------
# simultaneous SLAM + dynamics learnings
if !load_trained && (train_method == :EM_SLAM)
    with_alert("EM_SLAM training") do
        n_steps = is_quick_test ? 3 : max_train_steps + 1
        es = SEDL.EarlyStopping(; patience=40)
        obs_weight_schedule = if use_obs_weight_schedule
            step -> linear(1e-3, 1.0)(min(1.0, step / n_steps)) * max_obs_weight
        else
            step -> max_obs_weight
        end

        @smart_assert !use_simple_obs_model "Cannot use simple obs model with EM_SLAM"

        @info "EM_SLAM: Starting training..."
        SEDL.train_dynamics_EM_SLAM!(
            learned_motion_model,
            landmark_guess,
            landmarks_to_logpdf_obs,
            x0_dists,
            data_train.observations,
            data_train.controls,
            (; data_train.times, obs_frames);
            optimizer=adam,
            n_steps,
            n_particles,
            callback=em_callback(
                learned_motion_model,
                es;
                landmark_est=landmark_guess,
                n_steps,
                test_every=500,
            ),
            obs_weight_schedule,
        )
        @info "Best model: $(es.model_info)"
    end
    load_model_weights!()

    landmark_plt = let
        plot()
        SEDL.plot_2d_landmarks!(landmarks, "truth"; color=2)
        SEDL.plot_2d_trajectories!(data_train.states, "truth"; linecolor=1)
        SEDL.plot_2d_landmarks!(
            SEDL.landmarks_from_tensor(landmark_guess),
            "estimated";
            color=4,
            marker=:auto,
        )
    end
    display(landmark_plt)
    savefig(landmark_plt, joinpath(save_dir, "landmarks.pdf"))
end
##-----------------------------------------------------------
# train the model using supervised learning (assuming having ground truth trajectories)
function supervised_callback(
    learned_motion_model::BatchedMotionModel,
    early_stopping::SEDL.EarlyStopping;
    n_steps,
    test_every=100,
    plot_every=2_500,
)
    prog = Progress(n_steps; showspeed=true)

    function (r)
        Base.with_logger(logger) do
            @info "training" r.loss r.lr
        end

        # Compute test log_obs and plot a few trajectories.
        if r.step % test_every == 1 || r.step == n_steps
            valid_metrics = posterior_metrics(learned_motion_model, data_valid)
            to_stop = save_best_model(early_stopping, valid_metrics, (step=r.step,))

            Base.with_logger(logger) do
                @info "validation" valid_metrics... log_step_increment = 0
            end
        else
            to_stop = (; should_stop=false)
        end
        if r.step % plot_every == 1
            test_metrics = posterior_metrics(learned_motion_model, data_test)

            Base.with_logger(logger) do
                @info("testing", test_metrics..., log_step_increment = 0)
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? data_train : data_test
                plt = plot_posterior(
                    learned_motion_model,
                    sim_data,
                    id;
                    title="$name traj $id (iter $(r.step))",
                )

                Base.with_logger(logger) do
                    kv = [Symbol("traj_$id") => plt]
                    @info name log_step_increment = 0 kv...
                end
            end
        end

        next!(prog; showvalues=[(:loss, r.loss), (:step, r.step), (:learning_rate, r.lr)])
        to_stop
    end
end

function repeat_seq(seq::Vector{<:BatchTuple}, n::Integer)
    map(seq) do batch
        new_batches = BatchTuple[]
        for r in 1:(batch.batch_size)
            append!(new_batches, fill(batch[r], n))
        end
        nb = BatchTuple(new_batches)
        @smart_assert nb.batch_size == n * batch.batch_size
        nb
    end
end

simplifed_model = let
    core = SEDL.get_simplified_motion_core(scenario)
    BatchedMotionModel(tconf, sketch, core)
end
if train_method == :Handwritten
    learned_motion_model = simplifed_model
end

!load_trained &&
    (train_method ∈ [:FitTV, :FitTruth, :FitHand]) &&
    let
        data_multiplicity = (train_method == :FitHand) ? max(128 ÷ n_train_ex, 1) : 1
        states_train = if train_method == :FitTV
            est_result = SEDL.estimate_states_from_observations_SE2(
                scenario,
                obs_model,
                data_train.observations,
                data_train.states,
                data_train.Δt;
                observe_velocities=use_simple_obs_model,
                n_steps=is_quick_test ? 200 : 5000,
            )
            plot(est_result.loss_history; title="state estimation loss history") |> display
            est_result.states
        elseif train_method == :FitHand
            est_trajs = [BatchTuple[] for t in data_train.times]
            @info "Generating trajectories using the hand-written model..."
            for i in 1:n_train_ex
                pf_result = SEDL.batched_particle_filter(
                    repeat(data_train.states[1][i], 100_000),
                    (;
                        data_train.times,
                        obs_frames,
                        controls=getindex.(data_train.controls, i),
                        observations=getindex.(data_train.observations, i),
                    );
                    motion_model=simplifed_model,
                    logpdf_obs,
                    showprogress=false,
                )
                pf_trajs = SEDL.batched_trajectories(pf_result, data_multiplicity)
                push!.(est_trajs, pf_trajs)
            end
            map(est_trajs) do batches
                BatchTuple(specific_elems(batches))
            end
        elseif train_method == :FitTruth
            data_train.states
        else
            error("Unknown train_method: $train_method")
        end

        let
            traj_plt = plot(; title="Method: $train_method")
            SEDL.plot_2d_trajectories!(states_train, "Estimated states")
            SEDL.plot_2d_trajectories!(data_train.states, "True states"; linecolor=2)
            let
                @unzip xs, ys = landmarks
                scatter!(xs, ys; label="Landmarks")
            end
            display(traj_plt)
        end
        SEDL.plot_batched_series(
            data_train.times,
            getindex.(states_train, 1);
            truth=getindex.(data_train.states, 1),
            title="Training data",
        ) |> display

        core_in_set, core_out_set = SEDL.input_output_from_trajectory(
            learned_motion_model.sketch,
            states_train,
            repeat_seq(data_train.controls, data_multiplicity),
            data_train.times;
            # do not test if the graident is regularized
            test_consistency=train_method == :FitTruth,
        )

        @info "Number of training data: $(sum(x -> x.batch_size, core_in_set))"

        let
            total_steps = 40_000
            n_steps = is_quick_test ? 3 : total_steps + 1
            es = SEDL.EarlyStopping(; patience=40)
            SEDL.train_dynamics_supervised!(
                learned_motion_model.core,
                BatchTuple(core_in_set),
                BatchTuple(core_out_set),
                data_train.Δt;
                optimizer=adam,
                n_steps,
                callback=supervised_callback(
                    learned_motion_model, es; n_steps, test_every=200
                ),
            )
            @info "Best model: $(es.model_info)"
        end
        load_model_weights!()
    end
##-----------------------------------------------------------
# train the guide using variational inference
function vi_callback(
    learned_motion_model::BatchedMotionModel,
    early_stopping::SEDL.EarlyStopping;
    n_steps,
    trajs_per_ex=10,
    test_every=200,
)
    prog = Progress(n_steps; showspeed=true)
    test_x0_batch = repeat(data_test.states[1], trajs_per_ex)
    repeated_obs_seq = repeat.(data_test.observations, trajs_per_ex)
    repeated_control_seq = repeat.(data_test.controls, trajs_per_ex)

    function (r)
        Base.with_logger(logger) do
            @info "training" r.elbo r.obs_logp r.loss r.lr r.annealing
        end

        # Compute test elbo and plot a few trajectories.
        if r.step % test_every == 1 || r.step == n_steps
            test_trajs, test_lp_guide, test_core_in, test_core_out = guide(
                test_x0_batch, repeated_obs_seq, repeated_control_seq, data_test.Δt
            )
            test_lp_dynamics = SEDL.transition_logp(
                learned_motion_model.core, test_core_in, test_core_out, data_test.Δt
            )
            test_lp_obs = SEDL.observation_logp(obs_model, test_trajs, repeated_obs_seq)
            test_elbo =
                (test_lp_dynamics + test_lp_obs - sum(test_lp_guide)) /
                (length(test_trajs) * test_x0_batch.batch_size)

            Base.with_logger(logger) do
                @info "testing" elbo = test_elbo log_step_increment = 0
            end

            valid_metrics = posterior_metrics(learned_motion_model, data_valid)
            to_stop = save_best_model(early_stopping, valid_metrics, (; r.step))

            Base.with_logger(logger) do
                @info "validation" valid_metrics... log_step_increment = 0
            end

            for name in ["training", "testing"], id in [1, 2]
                sim_data = name == "training" ? data_train : data_test
                pf_plt = plot_posterior(
                    learned_motion_model,
                    sim_data,
                    id;
                    title="$name traj $id (iter $(r.step))",
                )

                guide_plt = SEDL.plot_guide_posterior(
                    guide, sim_data, id; title="$name traj $id (iter $(r.step))"
                )

                Base.with_logger(logger) do
                    kv = [Symbol("traj_$id") => guide_plt]
                    @info "$name/guide" log_step_increment = 0 kv...
                    kv = [Symbol("traj_$id") => pf_plt]
                    @info "$name/particle" log_step_increment = 0 kv...
                end
            end
        else
            to_stop = (; should_stop=false)
        end

        next!(
            prog;
            showvalues=[
                (:elbo, r.elbo),
                (:step, r.step),
                (:batch_size, r.batch_size),
                (:annealing, r.annealing),
                (:learning_rate, r.lr),
            ],
        )
        to_stop
    end
end

if !load_trained && train_method == :SVI
    with_alert("VI training") do
        n_steps = is_quick_test ? 3 : max_train_steps + 1

        n_repeat = min(10, 512 ÷ n_train_ex)
        vi_x0 = repeat(data_train.states[1], n_repeat)
        vi_obs_seq = repeat.(data_train.observations, n_repeat)
        vi_control_seq = repeat.(data_train.controls, n_repeat)

        @info "Training the guide..."
        es = SEDL.EarlyStopping(; patience=40)
        train_result = @time SEDL.train_VI!(
            guide,
            learned_motion_model.core,
            obs_model,
            vi_x0,
            vi_obs_seq,
            vi_control_seq,
            data_train.Δt;
            optimizer=adam,
            n_steps,
            # anneal_schedule=step -> linear(1e-3, 1.0)(min(1, step / n_steps)),
            callback=vi_callback(
                learned_motion_model, es; n_steps, test_every=500, trajs_per_ex=1
            ),
        )
        display(train_result)
        @info "Best model: $(es.model_info)"
        load_model_weights!()
    end
end
##-----------------------------------------------------------
plot_posterior(
    learned_motion_model, data_train; title="Posterior ($train_method learned)"
) |> display
plot_posterior(
    learned_motion_model,
    data_train;
    obs_frames=1:1,
    title="Open-loop prediction ($train_method learned)",
) |> display
if use_sim_data
    plot_posterior(
        motion_model, data_train; obs_frames=1:1, title="Open-loop prediction (true model)"
    ) |> display
end

plot_core_io(
    learned_motion_model, data_train; title="IO Posterior ($train_method learned)"
) |> display
plot_core_io(
    learned_motion_model,
    data_train;
    obs_frames=1:1,
    title="IO Open-loop prediction ($train_method learned)",
) |> display

function evaluate_model(motion_model, data_test)
    n_repeats = is_quick_test ? 3 : 10
    n_particles = is_quick_test ? 10_000 : 100_000
    metrics = posterior_metrics(motion_model, data_test; n_repeats, n_particles)
    # open_loop = posterior_metrics(motion_model, data_test; n_repeats, obs_frames=1:1).RMSE
    metrics
end

println("Testing dynamics model final performance...")
exp_name = isnothing(exp_name) ? string(train_method) : exp_name
test_performance = evaluate_model(learned_motion_model, data_test)
valid_performance = evaluate_model(learned_motion_model, data_valid)
perf_table = DataFrame([
    (; name=exp_name, test_performance...),
    (; name="(valid) $exp_name", valid_performance...),
])
display(perf_table)
CSV.write(joinpath(save_dir, "performance.csv"), perf_table)
##-----------------------------------------------------------