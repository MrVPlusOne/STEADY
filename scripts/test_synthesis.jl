# this script needs to be run inside the module SEDL
##
env = ComponentEnv()
components_scalar_arithmatic!(env)
env

x = Var(:x, PType(ℝ, PUnit(:L => 1)))
θ = Var(:θ, PType(ℝ, PUnit()))
result = bottom_up_enum(env, [x, θ], 3)
##
result.programs
