##-----------------------------------------------------------
# imports
using DrWatson
using Revise
using Flux: Flux
using CUDA: CUDA
CUDA.allowscalar(false)
using StatsPlots
StatsPlots.default(; dpi=300, legend=:outerbottom)
using TensorBoardLogger: TBLogger
using ProgressMeter

!true && begin
    include("../src/SEDL.jl")  # reloads the module
    using .SEDL
end
using SEDL: SEDL
using SEDL
##---------------------------------------------------------- 
# set up scenario and simulate some data
if !@isdefined is_quick_test
    is_quick_test = false
end
is_quick_test && @info "Quick testing VI..."

use_gpu = true
tconf = SEDL.TensorConfig(use_gpu, Float32)
device = use_gpu ? Flux.gpu : Flux.cpu

landmarks = [[10.0, 0.0], [-4.0, -2.0], [-6.0, 5.0]]
landmarks_tensor = landmarks |> SEDL.hcatreduce |> x -> Flux.cat(x'; dims=3) |> device
@smart_assert size(landmarks_tensor) == (length(landmarks), 2, 1)
size(landmarks_tensor)

linfo = SEDL.LandmarkInfo(; landmarks)
sce = SEDL.HovercraftScenario(linfo)

x0_batch = SEDL.BatchTuple(
    tconf,
    [
        (;
            pos=[0.5, 1.0] + 2randn(2),
            vel=[0.25 + 0.3randn(), 0.1 + 0.2randn()],
            θ=[π / 5 + π * randn()],
            ω=[π / 50 + 0.1π * randn()],
        ) for _ in 1:128
    ],
)

Δt = tconf(0.1)
times = 0:Δt:10
obs_frames = 1:10:length(times)

true_params = map(p -> convert(tconf.ftype, p), SEDL.simulation_params(sce))
sketch = SEDL.batched_sketch(sce)
motion_model = SEDL.BatchedMotionModel(tconf, sketch, SEDL.batched_core(sce, true_params))

# obs_model =
#     state -> SEDL.landmark_obs_model(state, (; landmarks=landmarks_tensor, linfo.σ_bearing))
obs_model = state -> SEDL.gaussian_obs_model(state, (pos=1.0, vel=1.0, θ=1.0, ω=1.0))

controller = let scontroller = SEDL.simulation_controller(sce)
    (args...) -> begin
        (; ul, ur) = scontroller(args...)
        SEDL.BatchTuple(tconf, [(ul=[ul], ur=[ur])])
    end
end

sim_en = SEDL.simulate_trajectory(times, x0_batch, (; motion_model, obs_model), controller)
# repeat_factor = 128
# sim_en = map(sim_en) do seq
#     map(b -> repeat(b, repeat_factor), seq)
# end

let
    first_states = [Flux.cpu(b[1].val) for b in sim_en.states]
    landmark_obs = [((; landmarks=fill(true, length(landmarks)))) for _ in first_states]
    obs_data = (; obs_frames, observations=landmark_obs)
    plot()
    SEDL.plot_2d_scenario!(first_states, obs_data, "Ground truth"; landmarks)
    SEDL.plot_2d_trajectories!(sim_en.states, "forward simulation") |> display
end

SEDL.plot_batched_series(times, TensorConfig(false).(sim_en.states)) |> display
##-----------------------------------------------------------
# set up the VI model
h_dim = 64
y_dim = sum(m -> size(m, 1), sim_en.observations[1].val)
guide = @time SEDL.mk_guide(;
    sketch,
    h_dim,
    y_dim,
    sample_states=(sim_en.states),
    sample_observations=(sim_en.observations),
    sample_controls=(sim_en.controls),
    Δt,
) |> device

log_joint =
    let u_seq = sim_en.controls, obs_seq = sim_en.observations, x1_truth = sim_en.states[1]
        x_seq -> (
            sum(
                map(x1_truth.val, x_seq[1].val) do x, y
                    SEDL.logpdf_normal(x, 0.01f0, y)
                end,
            ) +
            SEDL.transition_logp(motion_model, x_seq, u_seq, Δt) +
            SEDL.observation_logp(obs_model, x_seq, obs_seq)
        )
    end
# adam = Flux.Optimiser(Flux.ClipNorm(1.0), Flux.WeightDecay(1e-4), Flux.ADAM(1e-4))
adam = Flux.ADAM(1e-4)
save_dir = SEDL.data_dir(savename("test_vi", (; h_dim); connector="-"))
if isdir(save_dir)
    @warn "removing old data at $save_dir..."
    rm(save_dir, recursive=true)
end
logger = TBLogger(joinpath(save_dir, "tb_logs"))

elbo_history = []
let (prior_trajs, _) = guide(sim_en.observations, sim_en.controls, Δt)
    SEDL.plot_batched_series(
        times, SEDL.TensorConfig(false).(prior_trajs); title="Guide prior"
    ) |> display
end
##-----------------------------------------------------------
# train the guide
linear(from, to) = x -> from + (to - from) * x

total_steps = 5000
let n_steps = is_quick_test ? 2 : total_steps + 1, prog = Progress(n_steps; showspeed=true)
    test_n_traj = 100
    test_data = map([1, 10]) do i
        (
            id=i,
            test_true_states=map(b -> b[i], sim_en.states),
            test_obs=map(b -> repeat(b[i], test_n_traj), sim_en.observations),
            test_controls=map(b -> repeat(b[i], test_n_traj), sim_en.controls),
        )
    end

    callback =
        r -> begin
            push!(elbo_history, r.elbo)
            if r.step % 100 == 1
                foreach(test_data) do (; id, test_true_states, test_obs, test_controls)
                    test_trajs, _ = guide(test_obs, test_controls, Δt)
                    plt = SEDL.plot_batched_series(
                        times,
                        SEDL.TensorConfig(false).(test_trajs);
                        truth=test_true_states,
                        title="Trajectory $id (iter $(r.step))",
                    )
                    Base.with_logger(logger) do
                        kv = ["traj $id" => plt]
                        @info "training" kv...
                    end
                    display(plt)
                end
            end
            Base.with_logger(logger) do
                @info "training" r.elbo r.batch_size r.annealing
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
        end
    @info "Training the guide..."
    train_result = @time SEDL.train_guide!(
        guide,
        log_joint,
        sim_en.observations,
        sim_en.controls,
        Δt;
        optimizer=adam,
        n_steps,
        anneal_schedule=step -> linear(1e-3, 1.0)(min(1, 3step / n_steps)),
        callback,
        n_samples_f=step -> ceil(Int, linear(64, 256)(step / n_steps)),
        lr_schedule=let β = 100^2 / total_steps
            step -> 2e-3 / sqrt(β * step)
        end,
    )
    display(train_result)
end
##-----------------------------------------------------------
plot(elbo_history[500:end]; title="ELBO (training)") |> display