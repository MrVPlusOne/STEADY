# this script needs to be run inside the module SEDL
##
env = ComponentEnv()
components_scalar_arithmatic!(env)
components_vec2!(env)
components_transcendentals!(env)
env

l = Var(:l, PType(ℝ, PUnit(:L => 1)))
x = Var(:x, PType(ℝ2, PUnit(:L => 1)))
θ = Var(:θ, PType(ℝ, PUnit()))
result = bottom_up_enum(env, [x, l, θ], 8)
##
result.programs
