
specific_elems(xs) = identity.(xs)

@inline rotation2D(θ) = @SArray(
    [cos(θ) -sin(θ)
     sin(θ)  cos(θ)]
)

rotate2d(θ, v) = rotation2D(θ) * v

to_measurement(values) = begin
    μ = mean(values)
    σ = std(values)
    μ ± σ
end