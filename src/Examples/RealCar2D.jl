struct RealCarScenario <: Scenario end

Base.summary(io::IO, ::RealCarScenario) = print(io, "RealCarScenario")

function negate_angle_2d(angle_2d::AbsMat)
    @views vcat(angle_2d[1:1, :], -angle_2d[2:2, :])
end

function batched_sketch(::RealCarScenario)
    state_vars = (; pos=2, angle_2d=2, vel=2, ω=1)
    control_vars = (; twist_linear=3, twist_angular=3)
    input_vars = (; angle_2d=2, loc_v=2, ω=1, control_vars...)
    output_vars = (; a_loc=2, a_rot=1)

    state_to_input(x, u) =
        BatchTuple(x, u) do (; angle_2d, vel, ω), uval
            loc_v = rotate2d(negate_angle_2d(angle_2d), vel)
            (; angle_2d, loc_v, ω, uval...)
        end

    output_to_state(x, o, Δt) =
        BatchTuple(x, o) do (; pos, angle_2d, vel, ω), (; a_loc, a_rot)
            (
                pos=pos .+ vel .* Δt,
                angle_2d=rotate2d(ω * Δt, angle_2d),
                vel=vel .+ rotate2d(angle_2d, a_loc) .* Δt,
                ω=ω .+ a_rot .* Δt,
            )
        end

    output_from_state(x, x1, Δt) =
        BatchTuple(x, x1) do xv, x1v
            acc = (x1v.vel .- xv.vel) ./ Δt
            a_loc = rotate2d(negate_angle_2d(xv.angle_2d), acc)
            a_rot = (x1v.ω .- xv.ω) ./ Δt
            (; a_loc, a_rot)
        end

    BatchedMotionSketch(;
        state_vars,
        control_vars,
        input_vars,
        output_vars,
        state_to_input,
        output_to_state,
        output_from_state,
    )
end

function state_L2_loss_batched(::RealCarScenario)
    (state1::BatchTuple, state2::BatchTuple) -> map(state1.val, state2.val) do s1, s2
        sum((s1 .- s2) .^ 2, dims=1)
    end |> sum
end

module Dataset

using CSV
using DataFrames
using Interpolations: LinearInterpolation
using Quaternions
using ..SEDL
using ..SEDL: RealCarScenario

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

"""
Compute `angle2 - angle1` as a scalar.

## Examples
```
julia> angle_difference([cos(2.5) sin(2.5)], [cos(1.5) sin(1.5)])[1] ≈ 1.0
true
```
"""
function angle_difference(angle2::SEDL.AbsMat, angle1::SEDL.AbsMat)
    @. asin(angle1[:, 1:1] * angle2[:, 2:2] - angle1[:, 2:2] * angle2[:, 1:1])
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
        [
            ["twist.linear.x", "twist.linear.y", "twist.linear.z"] => :twist_linear,
            ["twist.angular.x", "twist.angular.y", "twist.angular.z"] => :twist_angular,
        ],
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

    ids1 = vcat(1:(T - 1), T - 1)
    ids2 = vcat(2:T, T)
    ω = angle_difference(pose_data.angle_2d[ids2, :], pose_data.angle_2d[ids1, :]) ./ ΔT
    speed_data = (; vel=(pose_data.pos[ids2, :] .- pose_data.pos[ids1, :]) ./ ΔT, ω)

    states = merge(pose_data, speed_data)
    batch_data = (
        times=map(tconf, times[1:(end ÷ n_trajs)]),
        states=break_trajectories(states, n_trajs, tconf),
        controls=break_trajectories(control_data, n_trajs, tconf),
    )
    batch_data
end

end