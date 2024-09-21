using Random, Distributions, Statistics


uniform_0_1 = Uniform(0, 1)


function normal_density(x::Real, μ::Real, σ::Real)
    return sqrt(2 * π * σ)^-1 * exp(
        -0.5 * (
            (x - μ) / σ
        )^2
    )
end


function std_normal_density(x::Real)
    return normal_density(x, 0, 1)
end


function metropolis_hastings_step(
    x::Real,
    proposal::Function,
    target_density::Function,
)
    # Propose a new position
    x_new = rand(proposal(x))

    # Get acceptance ratio
    α = target_density(x_new) / target_density(x)

    # Take step
    if rand(uniform_0_1) ≤ α
        return x_new, α
    else
        return x, α
    end
end


function metropolis_hastings_sampling(
    x_initial::Real,
    proposal::Function,
    target_density::Function,
    n_steps::Integer,
)
    # Initialise chain and vector of α values
    x_vals = zeros(n_steps)
    α_vals = zeros(n_steps)

    # Perform sampling
    x_vals[1] = x_initial
    for i in 2:n_steps
        x_vals[i], α_vals[i] = metropolis_hastings_step(
            x_vals[i-1], proposal, target_density
        )
    end
    return x_vals, α_vals
end


# Choices
PROPOSAL_SCALE = 1
N_STEPS = Int32(1e7)
proposal(x) = Uniform(x - PROPOSAL_SCALE, x + PROPOSAL_SCALE)
initial = 1.78

# Target
target = std_normal_density

# Do sampling
chain, a_frac = metropolis_hastings_sampling(initial, proposal, target, N_STEPS)

print(mean(chain), "\n")
print(std(chain))