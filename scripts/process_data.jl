using CSV
using DataFrames
using Plots
Plots.default(; dpi=300, legend=:outerbottom)
using Interpolations: LinearInterpolation
using Quaternions

!true && begin
    include("../src/SEDL.jl")
    using .SEDL
end
using SEDL
using .SEDL: @kwdef
##-----------------------------------------------------------
# utilities
° = SEDL.°
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
Resample the data frame data to a named tuple of component matrices of size (T, dims). 
"""
function resample_data(log, times, schema::Vector{Pair{Vector{String},Symbol}})
    NamedTuple(
        map(schema) do (cols, new_name)
            new_name =>
                map(cols) do c
                    LinearInterpolation(log[:, :Time], log[:, c]).(times)
                end |> hcatreduce
        end,
    )
end

function gaussian_observations(states::NamedTuple)
    map(states) do vec
        vec + randn(eltype(vec), size(vec)) * 0.5
    end
end

"""
Each component of `traj` should be of shape `[n_times, n_dims]`.
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

function read_data_from_csv(data_dir, tconf::TensorConfig)
    all_data =
        pose_data, control_data = [
            CSV.read(joinpath(data_dir, "$name.csv"), DataFrame) for
            name in ["vrpn_client_node-alpha_truck-pose", "vesc_drive"]
        ]

    sec_start = maximum(l -> l[1, "header.stamp.secs"], all_data)

    foreach(log -> normalize_times!(log, sec_start), all_data)

    time_range = (
        maximum(l -> l[1, "Time"], all_data), minimum(l -> l[end, "Time"], all_data)
    )
    n_trajs = floor(Int, time_range[2] - time_range[1]) ÷ 10
    times = LinRange(time_range[1], time_range[2], n_trajs * 100)
    T = length(times)
    ΔT = times[2] - times[1]

    pose_data = resample_data(
        pose_data,
        times,
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

    ids1 = vcat(1:(T - 1), T - 1)
    ids2 = vcat(2:T, T)
    ω_2d = (pose_data.angle_2d[ids2, :] .- pose_data.angle_2d[ids1, :]) ./ ΔT
    speed_data = (; vel=(pose_data.pos[ids2, :] .- pose_data.pos[ids1, :]) ./ ΔT, ω_2d)

    control_data = resample_data(
        control_data,
        times,
        [
            ["twist.linear.x", "twist.linear.y", "twist.linear.z"] => :twist_linear,
            ["twist.angular.x", "twist.angular.y", "twist.angular.z"] => :twist_angular,
        ],
    )

    states = merge(pose_data, speed_data)
    batch_data = (
        times=map(tconf, times[1:(end ÷ n_trajs)]),
        states=break_trajectories(states, n_trajs, tconf),
        observations=break_trajectories(gaussian_observations(pose_data), n_trajs, tconf),
        controls=break_trajectories(control_data, n_trajs, tconf),
    )
    batch_data
end
##-----------------------------------------------------------
# train_data = read_data_from_csv(SEDL.data_dir("real_data", "simple_loop"))
# test_data = read_data_from_csv(SEDL.data_dir("real_data", "simple_loop_test"))
# ##-----------------------------------------------------------
# SEDL.plot_batched_series(train_data.times, getindex.(train_data.states, 2)) |> display
