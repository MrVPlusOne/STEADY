"""
The (batched) 2D hovercraft scenario.
"""
@kwdef(struct HovercraftScenarioB{LI<:LandmarkInfo} <: Scenario
    landmark_info::LI
end)

function sample_initial_state(::HovercraftScenario, x0)
    DistrIterator((
        pos=SMvNormal(x0.pos, 0.2),
        vel=SMvNormal(x0.vel, 0.2),
        θ=CircularNormal(x0.θ, 8°),
        ω=Normal(x0.ω, 0.1),
    ))
end