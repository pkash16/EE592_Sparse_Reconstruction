using LinearAlgebra
using IterativeSolvers
using LinearOperators

function updateZ(Zk, D, X, λ)
    #this is iterative soft thresholding for each column of Z.

    #initialization? Zero out Zk. TODO figure out if this is necessary
    Zk = zeros(size(Zk))

    #no need to recompute α, so we compute it first here.
    α = opnorm(D)^2
    Dhd = D'D;

    #utility functions for the inner loop
    sgn(x) = x == 0 ? 0 : x / abs(x)

    for i = 1:size(Zk,2)
        Zki = Zk[:,i]
        #iterate for the size of the patches (I think this is required to resach convergence?)
        for iter = 1:size(D,2)
            Yki = Zki - (1/α) * (Dhd*Zki - D'*X[:,i])
            Zki = sgn.(Yki) .* max.(abs.(Yki) .- (λ / (2*α)), 0)
        end

        Zk[:,i] = Zki
    end

    return Zk
end

function updateYk(dk, pk)
    tmp = dk + pk
    if (norm(tmp)^2) <= 1
        return tmp
    else
        return tmp / (norm(tmp)^2)
    end
end

function updateD(Dk, Z, X, μ)
    #initialization zero out Dk TODO: findout if this is necessary
    D̃ = zeros(size(Dk'))
    X̃ = X'

    for i = 1:size(D̃, 2)

        dk = D̃[:,i]
        pk = zeros(size(dk))
        yk = zeros(size(dk))

        for iter = 1:10
            b = [X̃[:,i]; yk - pk]
            combined = [Z'; sqrt(μ/2) * I(length(dk))]

            AhA = combined' * combined
            Ahb = combined' * b

            dk = zeros(size(dk))
            cg!(dk, Matrix(AhA), Vector(Ahb), maxiter=200)
            yk = updateYk(dk, pk)
            pk = pk - (dk - yk)
        end

        D̃[:,i] = dk
    end

    return D̃'
end

#takes in input image time series [W,H, time] and converts to patches.
#patches will be of size patch_size x patch_size
#patches will be overlapped with overlap # pixels
function patchify(image, patch_size, overlap)
    for time in size(image,3)
            
    end
end
