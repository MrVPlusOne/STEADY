# This implements the simple landmark SLAM example from the GTSAM tutorial
# https://gtsam.org/tutorials/intro.html#magicparlabel-65635 using Turing.jl.

using DrWatson
@quickactivate "ProbPRL"

using Turing
using Optim
using StatsPlots
using LabelledArrays
using StatsBase
using DataFrames
using Statistics: norm

get_pos(state) = state[1:2]
get_θ(state) = state[3]
const ° = π/180

@inline rotation2D(α) = 
    [cos(α) -sin(α)
     sin(α)  cos(α)]

function bearing_range_dist(target, state)
    rel = target - get_pos(state)
    MvNormal([atan(rel[2], rel[1]) - get_θ(state), norm(rel)], [0.1, 0.2])
end

function odometry_dsit(x1, x2)
    rel = x2 - x1
    MvNormal([rotation2D(-get_θ(x1)) * get_pos(rel); rel[3]], [0.3, 0.3, 0.1])
end

@model function slam_model(odometry_readings, br_readings)
    local x1, x2, x3, l1, l2
    # initial pose
    x1 ~ MvNormal([0., 0., 0.], [0.3, 0.3, 0.1])
    # pose prior (uninformative)
    x2 ~ MvNormal([0., 0., 0.], 100 * ones(3))
    x3 ~ MvNormal([0., 0., 0.], 100 * ones(3))
    
    # odometry observations
    odometry_readings[1] ~ odometry_dsit(x1, x2)
    odometry_readings[2] ~ odometry_dsit(x2, x3)

    # landmark prior
    l1 ~ MvNormal([0., 0.], [100., 100.])
    l2 ~ MvNormal([0., 0.], [100., 100.])

    # range/bearing observations
    br_readings[1] ~ bearing_range_dist(l1, x1)
    br_readings[2] ~ bearing_range_dist(l1, x2)
    br_readings[3] ~ bearing_range_dist(l2, x3)
end

model = let
    odometry_data = [[2., 0., 0.0], [2., 0., 0.]]
    br_data = [[45°, sqrt(8.)], [90°, 2.], [90°, 2.]]
    slam_model(odometry_data, br_data)
end

println("Sampling with NUTS...")
chain = @time Turing.sample(model, NUTS(0.7), 500)

println("Performing MAP Optimization...")
r = @time optimize(model, MAP(), autodiff=:forwarddiff)
println("Computing information matrix...")
infomat = @time informationmatrix(r)

function cov_mat(var_names) 
    ids = collect(Symbol.(var_names))
    infomat[ids, ids]
end

p = plot()
for t in 1:3
    scatter!(chain["x$t[1]"], chain["x$t[2]"], label="x$t")
end
for i in 1:2
    scatter!(chain["l$i[1]"], chain["l$i[2]"], label="l$i")
end
p_nuts = plot(p, aspect_ratio=1, title="NUTS Samples")

p = plot()
for t in 1:3
    dist = fit(MvNormal, [chain["x$t[1]"]'; chain["x$t[2]"]'])
    covellipse!(dist.μ, dist.Σ, label="x$t")
end
for t in 1:2
    dist = fit(MvNormal, [chain["l$t[1]"]'; chain["l$t[2]"]'])
    covellipse!(dist.μ, dist.Σ, label="l$t")
end
p_nuts_fit = plot(p, aspect_ratio=1, title="NUTS Fit")

p = plot()
for t = 1:3
    labels = Symbol.(["x$t[1]", "x$t[2]"])
    μ = r.values[labels]
    covellipse!(μ, cov_mat(labels), label="x$t")
end
for t = 1:2
    labels = Symbol.(["l$t[1]", "l$t[2]"])
    μ = r.values[labels]
    covellipse!(μ, cov_mat(labels), label="l$t")
end
p_map_fit = plot(p, aspect_ratio=1, title="MAP Fit")

display.([p_nuts, p_nuts_fit, p_map_fit])