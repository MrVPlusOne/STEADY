# this script needs to be run inside the module SEDL
##
env = ComponentEnv()
components_scalar_arithmatic!(env)
components_vec2!(env)
components_transcendentals!(env)
env

l = Var(:l, ℝ, :L => 1)
x = Var(:x, ℝ2, :L => 1)
v = Var(:v, ℝ2, :L => 1, :T => -1)
θ = Var(:θ, ℝ)
steer = Var(:steer, ℝ, :T => -1)
wall_x = Var(:wall_x, ℝ, :L => 1)

derivative(x)

result = bottom_up_enum(env, [x, l, θ], 7)
##

n_steps = 10
vdata = VariableData(
    states = Dict([
        x => (MvNormal(zeros(2), ones(2)*0.1), MvNormal(zeros(2), ones(2))),
        θ => (Normal(pi/2, 0.1), Normal(0.0, 0.5)),
    ]),
    actions = Dict([
        steer => fill(1.0, n_steps),
    ]),
    dynamics_params = Dict([
        l => Uniform(0.0, 2.0),
    ]),
    others = Dict([
        wall_x => Uniform(0.0, 50.0)
    ]),
)
map_synthesis(env, vdata, identity, identity, 4)
##