struct RealCarScenario <: Scenario end

Base.summary(io::IO, ::RealCarScenario) = print(io, "RealCarScenario")

function batched_sketch(::RealCarScenario)
    control_vars = (; twist_linear=1, twist_angular=1)
    batched_sketch_SE2(control_vars)
end

function state_L2_loss_batched(::RealCarScenario)
    L2_in_SE2_batched
end

function pose_to_opt_vars(::RealCarScenario)
    pose_to_opt_vars_SE2
end

function pose_from_opt_vars(::RealCarScenario)
    pose_from_opt_vars_SE2
end

function get_simplified_motion_core(::RealCarScenario)
    twist_linear_scale = 1.0f0
    twist_angular_scale = 0.5f0
    max_a_linear = 6.0f0
    max_a_angular = 6.0f0

    acc_σs = (a_loc=5.0f0, a_rot=2.0f0)

    (core_input::BatchTuple, Δt) -> begin
        local μs = BatchTuple(core_input) do (; twist_linear, twist_angular, loc_v, ω)
            â = (twist_linear_scale * twist_linear .- loc_v[1:1, :]) ./ Δt
            a_x = @. ifelse(abs(â) < max_a_linear, â, sign(â) .* max_a_linear)
            a_y = @. -loc_v[2:2, :] / Δt
            a_loc = vcat(a_x, a_y)

            â_rot = (twist_angular_scale * twist_angular .- ω) ./ Δt
            a_rot = @. ifelse(
                abs(â_rot) < max_a_angular, â_rot, sign(â_rot) .* max_a_angular
            )
            (; a_loc, a_rot)
        end
        local σs = BatchTuple(core_input) do _
            acc_σs
        end
        (; μs, σs)
    end
end

module Dataset

using CSV
using DataFrames
using Interpolations: LinearInterpolation
using Quaternions
using ..SEDL
using ..SEDL: RealCarScenario
using Flux
using ProgressMeter
using NoiseRobustDifferentiation: NoiseRobustDifferentiation as NDiff


function find_time_range(logs)
    (
        t_start=maximum(l -> l[1, "header.stamp.secs"], logs),
        t_end=minimum(l -> l[end, "header.stamp.secs"], logs),
    )
end

function get_normalized_times(log, t_start)
    map(log[:, "header.stamp.secs"], log[:, "header.stamp.nsecs"]) do s, ns
        s - t_start + ns / 1e9
    end
end

function normalize_times!(log, t_start)
    log[:, :Time] = get_normalized_times(log, t_start)
end

"""
Turn the data frame into a named tuple with component matrices of size `(times, dims)`. 
"""
function regroup_data(df::DataFrame, schema::Vector{Pair{Vector{String},Symbol}})
    NamedTuple(
        map(schema) do (cols, new_name)
            new_name => map(cols) do c
                df[:, c]
            end |> hcatreduce
        end,
    )
end

"""
Resample data using linear interpolation.
"""
function resample_data(data::NamedTuple, old_times, new_times)
    map(data) do xs
        map(1:size(xs, 2)) do c
            LinearInterpolation(old_times, xs[:, c]).(new_times)
        end |> hcatreduce
    end
end

"""
Each component of `traj` should be of size `(times, dims)`.
"""
function break_trajectories(traj::NamedTuple, n_traj::Int, tconf::TensorConfig)
    T = size(traj[1], 1)
    T′ = T ÷ n_traj
    map(1:T′) do t
        BatchTuple(
            tconf,
            n_traj,
            map(traj) do comp
                tconf(comp[t:T′:T, :]')
            end,
        )
    end
end

function quat_to_angle(x, y, z, w)
    quat = Quaternion(w, x, y, z)
    vcat(axis(quat), angle(quat))
end

function states_from_poses_finitediff(
    poses::Vector{<:BatchTuple}, Δt::Real
)::Vector{<:BatchTuple}
    T = length(poses)
    ids1 = vcat(1:(T - 1), T - 1)
    ids2 = vcat(2:T, T)
    map(poses, poses[ids1], poses[ids2]) do p, p1, p2
        BatchTuple(p, p1, p2) do val, val1, val2
            ω = SEDL.angle_2d_diff(val2.angle_2d, val1.angle_2d; dims=1) ./ Δt
            vel = (val2.pos .- val1.pos) ./ Δt
            (; val.pos, val.angle_2d, vel, ω)
        end
    end
end

"""
Estimate state derivatives using total-variation regularization.
"""
function states_from_poses_tv(poses::Vector{<:BatchTuple}, Δt::Real; α, n_iters)
    T = length(poses)
    tconf = poses[1].tconf
    tconf_cpu = Flux.cpu(tconf)
    batch_size = poses[1].batch_size
    function compute_derivatives(values)
        NDiff.tvdiff(values, n_iters, α; dx=Δt, diff_kernel="square")
    end
    function estimate_single(poses::Vector{<:BatchTuple})
        @smart_assert poses[1].batch_size == 1
        @smart_assert poses[1].tconf.on_gpu == false
        vel1 = compute_derivatives([p.val.pos[1, 1] for p in poses])
        vel2 = compute_derivatives([p.val.pos[2, 1] for p in poses])
        dθ = map(1:(T - 1)) do t
            SEDL.angle_2d_diff(poses[t + 1].val.angle_2d, poses[t].val.angle_2d; dims=1)[1]
        end
        ω = compute_derivatives(
            [zero(eltype(dθ)); accumulate(+, dθ; init=zero(eltype(dθ)))]
        )
        map(1:T) do t
            val = poses[t].val
            BatchTuple(
                tconf_cpu,
                1,
                (
                    pos=val.pos,
                    angle_2d=val.angle_2d,
                    vel=tconf_cpu(vcat([vel1[t];;], [vel2[t];;])),
                    ω=[tconf_cpu(ω[t]);;],
                ),
            )
        end
    end

    poses_cpu = Flux.cpu.(poses)
    prog = Progress(batch_size; desc="states_from_poses_tv")
    seqs = map(1:batch_size) do i
        est = estimate_single(getindex.(poses_cpu, i))
        next!(prog)
        est
    end
    map(1:T) do t
        BatchTuple(tconf.(getindex.(seqs, t)))
    end
end

"""
Read the trajectory data from the csv files generated by rospy.
"""
function read_data_from_csv(::RealCarScenario, data_dir, tconf::TensorConfig)
    all_data =
        pose_data, control_data = [
            CSV.read(joinpath(data_dir, "$name.csv"), DataFrame) for
            name in ["vrpn_client_node-alpha_truck-pose", "vesc_drive"]
        ]

    sec_start = maximum(l -> l[1, "header.stamp.secs"], all_data)

    foreach(log -> normalize_times!(log, sec_start), all_data)

    pose_times = pose_data[:, :Time]
    pose_data = regroup_data(
        pose_data,
        [
            ["pose.position.x", "pose.position.y"] => :pos,
            [
                "pose.orientation.x",
                "pose.orientation.y",
                "pose.orientation.z",
                "pose.orientation.w",
            ] => :quat,
        ],
    )

    angle_2d = map(eachrow(pose_data.quat)) do r
        θ = 2.0 * atan(r[3], r[4])
        [cos(θ) sin(θ)]
    end |> vcatreduce
    pose_data = (; pose_data.pos, angle_2d)

    control_times = control_data[:, :Time]
    control_data = regroup_data(
        control_data,
        [["twist.linear.x"] => :twist_linear, ["twist.angular.z"] => :twist_angular],
    )

    time_range = (
        maximum(l -> l[1, "Time"], all_data), minimum(l -> l[end, "Time"], all_data)
    )
    n_trajs = floor(Int, time_range[2] - time_range[1]) ÷ 10
    times = LinRange(time_range[1], time_range[2], n_trajs * 100)
    T = length(times)
    ΔT = times[2] - times[1]

    pose_data = resample_data(pose_data, pose_times, times)
    control_data = resample_data(control_data, control_times, times)
    (; angle_2d, pos) = pose_data

    ids1 = vcat(1:(T - 1), T - 1)
    ids2 = vcat(2:T, T)
    ω = SEDL.angle_2d_diff(angle_2d[ids2, :], angle_2d[ids1, :]; dims=2) ./ ΔT
    speed_data = (; vel=(pos[ids2, :] .- pos[ids1, :]) ./ ΔT, ω)

    states = merge(pose_data, speed_data)
    batch_data = (
        times=map(tconf, times[1:(end ÷ n_trajs)]),
        states=break_trajectories(states, n_trajs, tconf),
        controls=break_trajectories(control_data, n_trajs, tconf),
    )
    batch_data
end

"""
Note that the first state will not be optimized.
"""
function estimate_states_from_observations(
    sce::RealCarScenario,
    obs_model,
    obs_seq,
    true_states,
    Δt;
    n_steps=5000,
    showprogress=true,
)
    T = length(true_states)
    states_guess = BatchTuple(true_states[2:end])
    observations = BatchTuple(obs_seq[2:end])
    poses_guess = BatchTuple(states_guess) do (; pos, angle_2d)
        (; pos, angle_2d)
    end
    optimizer = ADAM()
    loss_history = []
    prog = Progress(n_steps; desc="estimate_states_from_observations", enabled=showprogress)
    callback = info -> begin
        push!(loss_history, info.loss)
        next!(prog)
        (; should_stop=false)
    end
    poses_est::BatchTuple = SEDL.optimize_states_from_observations(
        sce,
        obs_model,
        observations,
        poses_guess;
        optimizer,
        n_steps,
        lr_schedule=nothing,
        callback,
    )
    pose0 = BatchTuple(true_states[1]) do (; pos, angle_2d)
        (; pos, angle_2d)
    end
    poses = vcat(pose0, split(poses_est, T - 1))
    states = states_from_poses_tv(poses, Δt; α=0.01, n_iters=n_steps ÷ 5)
    (; states, loss_history)
end

end