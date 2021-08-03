module Car1D

# TODO: refactor using multiple dispatch

using UnPack
using Turing
using OrdinaryDiffEq
# using DiffEqSensitivity
# using Zygote: Buffer

@enum StateIndices Pos Vel
Base.to_index(i::StateIndices) = Int(i)+1
new_state(;pos, vel) = [pos, vel]
@inline dim_state() = 2
@inline dim_action() = 1
new_action(;force) = [force]

function dynamics(x::X, p, t)::X where X
    f, drag, inv_mass = p
    v = x[Vel]
    d_pos = v
    d_vel = (f - drag * v) * inv_mass
    [d_pos, d_vel]
end

function dynamics!(du, x, p, t)
    f, drag, inv_mass = p
    v = x[Vel]
    du[Pos] = v
    du[Vel] = (f - drag * v) * inv_mass
end

function mk_motion_model(x0, u0, (t0, tf) ; drag, mass)
    params = [u0[1], drag, 1.0/mass]
    prob = ODEProblem{true}(dynamics!, x0, (t0, tf), params)
    integrator = init(
        prob, BS3(),
        dtmin=0.01, force_dtmin=true, # we want to guarantee good solver speed
        dt=0.1,
        abstol=1e-3, save_everystep=false,
    )

    (x, u, t::Float64, t1::Float64) -> begin
        integrator.p[1] = u[1]
        set_u!(integrator, x)
        step!(integrator, t1-t, true)
        integrator.u
    end
end

# using ..NumericalIntegration
# function mk_motion_model(x0, u0, (t0, tf) ; drag, mass)
#     (x, u, t::Float64, t1::Float64) -> begin
#         params = [u[1], drag, 1.0/mass]
#         integrate_forward(dynamics, x, params, (t, t1), Euler, 1)
#     end
# end

function wall_dist()
    Uniform(0., 50.)
end

function motion_noise(x::AbstractVector, Δt)
    MvNormal(x, new_state(pos=0.1, vel=0.2) .* (0.1 + abs(x[Vel]) .* Δt ))
end

function sensor_dist(x, wall_pos; noise_scale)
    sensor_max_range = 5.0
    d = if 0 <= wall_pos - x[Pos] <= sensor_max_range
            wall_pos - x[Pos]
        else
            sensor_max_range + (wall_pos - x[Pos])*1e-4
        end
    Normal(d, 0.2noise_scale)
end

function speed_dist(x; noise_scale)
    Normal(x[Vel], 0.2noise_scale)
end

function odometry_dist(x, x1; noise_scale)
    Δ = x1[Pos] - x[Pos]
    Normal(Δ, 0.1noise_scale * (1+abs(Δ)))
end

function controller(vel, sensor)
    stop_dis = 2.0
    max_force = 10.0
    is_stopping = sensor < stop_dis
    target_v = is_stopping ? 0.0 : 2.0
    k = is_stopping ? 5.0 : 1.0
    clamp((target_v - vel) * k, -max_force, max_force)
end

@model function data_process(
    times;
    dyn_disturbance::Bool,
    noise_scale::Float64=1.0,
    s0_dist=MvNormal([0., 0.], [0.1, 0.1]),
    s0=missing,
    drag=missing, mass=missing, wall_pos=missing,
    actions=missing, 
    odometry_readings=missing, sensor_readings=missing, speed_readings=missing, 
)
    drag ~ Uniform(0.0, 1.0)
    mass ~ Uniform(0.5, 5)
    
    steps = length(times)-1
    no_actions = ismissing(actions)

    wall_pos ~ wall_dist()
    Num = typeof(wall_pos)
    ismissing(actions) && (actions = Matrix{Num}(undef, dim_action(), steps))
    ismissing(odometry_readings) && (odometry_readings = Vector{Num}(undef, steps))
    ismissing(sensor_readings) && (sensor_readings = Vector{Num}(undef, steps))
    ismissing(speed_readings) && (speed_readings = Vector{Num}(undef, steps))
        
    s0 ~ s0_dist
    states = Matrix{Num}(undef, 2, steps)
    u0 = new_action(force=0.0)
    states[:, 1] .= s0
    motion_model = mk_motion_model(s0, u0, (times[1], times[end]) ;drag, mass)
    for i in 1:steps
        x = states[:, i]
        x_last = i == 1 ? x : states[:, i-1]

        speed_readings[i] ~ speed_dist(x; noise_scale)
        sensor_readings[i] ~ sensor_dist(x, wall_pos; noise_scale)
        odometry_readings[i] ~ odometry_dist(x_last, x; noise_scale)

        no_actions && (actions[:, i] .= controller(speed_readings[i], sensor_readings[i]))

        t, t1 = times[i], times[i+1]
        x̂ = motion_model(x, actions[:, i], t, t1)
        if i < steps
            if dyn_disturbance
                states[:, i+1] ~ motion_noise(x̂, t1-t)
            else
                states[:, i+1] = x̂
            end
        end
    end
    data = (;actions, odometry_readings, sensor_readings, speed_readings)
    return (;times, states, wall_pos, data)
end

function posterior_density(
    (drag, mass, wall_pos, states),
    times,
    (actions, odometry_readings, sensor_readings, speed_readings);
)
    score = 0.0
    ~(x, dist) = score += logpdf(dist, x)

    drag ~ Uniform(0.0, 1.0)
    mass ~ Uniform(0.5, 5)
    wall_pos ~ wall_dist()
    states[:, 1] ~ MvNormal(new_state(pos=0.0, vel=0.0), new_state(pos=0.1, vel=0.1))

    steps = length(times)-1
        
    motion_model = 
        mk_motion_model(states[:, 1], actions[:, 1], (times[1], times[end]) ;drag, mass)

    for i in 1:steps
        x = states[:, i]
        x_last = states[:, max(i-1, 1)]

        t, t1 = times[i], times[i+1]
        speed_readings[i] ~ speed_dist(x)
        sensor_readings[i] ~ sensor_dist(x, wall_pos)
        odometry_readings[i] ~ odometry_dist(x_last, x)
        x̂ = motion_model(x, actions[:, i], t, t1)
        if i < steps
            states[:, i+1] ~ motion_noise(x̂, t1-t)
            # states[:, i+1] = x̂
        end
    end

    return score
end

end # module Car1D