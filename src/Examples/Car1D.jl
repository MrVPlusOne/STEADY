module Car1D

# TODO: refactor using multiple dispatch

using UnPack
using Turing
using OrdinaryDiffEq

@enum StateIndices Pos Vel AccF
Base.to_index(i::StateIndices) = Int(i)+1
new_state(;pos, vel) = [pos, vel]
@inline dim_state() = 2
@inline dim_action() = 1

function dynamics!(du, x, p, t)
    @unpack f, drag, inv_mass = p
    v = x[Vel]
    du[Pos] = v
    du[Vel] = (f - drag * v) * inv_mass
end

const integrator_cache = Ref{Union{Any}}(nothing)
function mk_motion_model(x0, u0, (t0, tf) ; drag, mass)
    if integrator_cache[] === nothing
        params = (; f = u0[1], drag, inv_mass=1.0/mass)
        prob = ODEProblem(dynamics!, x0, (t0, tf), params)
        # integrator = init(prob, BS3(); save_everystep=false, abstol=1e-3, dt)
        integrator = init(
            prob, BS3(),
            dtmin=0.01, force_dtmin=true, # we want to guarantee good solver speed
            dt=0.1,
            abstol=1e-3, save_everystep=false,
        )
        integrator_cache[] = integrator
    else
        integrator = integrator_cache[]
        reinit!(integrator, x0)
    end
    (x, u, t::Float64, t1::Float64) -> begin
        if any(isnan.(x)) || any(isinf.(x))
            error("Bad state value: $x")
        end
        integrator.p = (; f = u[1], drag, inv_mass=1.0/mass)
        step!(integrator, t1-t, true)
        integrator.u
    end
end

function mk_motion_model_euler(; drag, mass)
    (x, u, t0::Float64, t1::Float64) -> begin
        if any(isnan.(x)) || any(isinf.(x))
            error("Bad state value: $x")
        end
        params = (; f = u[1], drag, inv_mass=1.0/mass)
        n_steps = ceil(Int, (t1-t0)/0.002)
        dt = (t1-t0)/n_steps
        t = t0
        s = copy(x)
        ds = zero(x)
        for _ in 1:n_steps
            dynamics!(ds, s, params, t)
            ds .*= dt
            s .+= ds
            t += dt
        end
        s
    end
end

# function motion_noise(x)
#     MvNormal(x, new_state(pos=0.1, vel=0.2) * abs(x[Vel]))
# end

function sensor_dist(x, wall_pos)
    sensor_max_range = 5.0
    truncated(Normal(wall_pos - x[Pos], 0.2), 0., sensor_max_range)
end

function speed_dist(x)
    Normal(x[Vel], 0.2)
end

function odometry_dist(x, x1)
    Δ = x1[Pos] - x[Pos]
    Normal(Δ, 0.1+0.1*abs(Δ))
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
    euler = false,
    drag=missing, mass=missing, wall_pos=missing,
    actions=missing, 
    odometry_readings=missing, sensor_readings=missing, speed_readings=missing, 
)
    drag ~ Uniform(0.0, 1.0)
    mass ~ Uniform(0.5, 5)

    motion_model = 
        euler ? mk_motion_model_euler(; drag, mass) : mk_motion_model(;drag, mass)
    
    steps = length(times)-1
    no_actions = ismissing(actions)

    wall_pos ~ truncated(Normal(5.0, 5.0), 4.0, 50.)
    Num = typeof(wall_pos)
    ismissing(actions) && (actions = Matrix{Num}(undef, dim_action(), steps))
    ismissing(odometry_readings) && (odometry_readings = Vector{Num}(undef, steps))
    ismissing(sensor_readings) && (sensor_readings = Vector{Num}(undef, steps))
    ismissing(speed_readings) && (speed_readings = Vector{Num}(undef, steps))
        
    states = Matrix{Num}(undef, 2, steps)
    states[:, 1] = new_state(pos=0., vel=0.)
    for i in 1:steps
        x = states[:, i]
        x_last = i == 1 ? x : states[:, i-1]

        speed_readings[i] ~ speed_dist(x)
        sensor_readings[i] ~ sensor_dist(x, wall_pos)
        odometry_readings[i] ~ odometry_dist(x_last, x)

        no_actions && (actions[:, i] .= controller(speed_readings[i], sensor_readings[i]))

        t, t1 = times[i], times[i+1]
        x̂ = motion_model(x, actions[:, i], t, t1)
        if i < steps
            # states[:, i+1] ~ motion_noise(x̂)
            states[:, i+1] = x̂
        end
    end
    data = (;actions, odometry_readings, sensor_readings, speed_readings)
    return (;times, states, wall_pos, data)
end

@model function infer_process(
    times;
    euler = false,
    actions, odometry_readings, sensor_readings, speed_readings, 
)
    drag ~ Uniform(0.0, 1.0)
    mass ~ Uniform(0.5, 5)
    wall_pos ~ truncated(Normal(5.0, 5.0), 4.0, 50.)

    motion_model = 
        euler ? mk_motion_model_euler(; drag, mass) : mk_motion_model(;drag, mass)

    steps = length(times)-1

    Num = typeof(wall_pos)
        
    states = Matrix{Num}(undef, 2, steps)
    states[:, 1] = new_state(pos=0., vel=0.)
    for i in 1:steps
        x = states[:, i]

        t, t1 = times[i], times[i+1]
        x̂ = motion_model(x, actions[:, i], t, t1)
        if i < steps
            # states[:, i+1] ~ motion_noise(x̂)
            states[:, i+1] = x̂
        end
    end
    cols = collect(eachcol(states))
    speed_readings .~ speed_dist.(cols)
    sensor_readings .~ sensor_dist.(cols, wall_pos)
    odometry_readings[1] ~ odometry_dist(cols[1], cols[1])
    odometry_readings[2:end] .~ odometry_dist.(cols[1:end-1], cols[2:end])
    return nothing
end

end # module Car1D