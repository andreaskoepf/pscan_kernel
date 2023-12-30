import torch
from barrel_pscan import pscan_scan_fwd
from time import perf_counter_ns

def expand_(A, X):
    if A.size(1) == 1:
        return
    #print("expand_ ", A.shape, X.shape)
    T = 2 * (A.size(1) // 2)
    Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
    Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
    #print("Aa, Xa", Aa.shape, Xa.shape)
    Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
    Aa[:, :, 1].mul_(Aa[:, :, 0])
    expand_(Aa[:, :, 1], Xa[:, :, 1])
    Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
    Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
    if T < A.size(1):
        #print('fixup:', T, A.size(1))
        X[:, -1].add_(A[:, -1].mul(X[:, -2]))
        A[:, -1].mul_(A[:, -2])


#N, T, D = 2, 1047,
#  3
def benchmark():
    #N, T, D = 26, 16, 32
    #N, T, D = 128, 1024, 32
    #N, T, D = 256, 2048, 256
    N, T, D = 384, 1024, 64
    #N, T, D = 1024, 1024, 100

    #A = torch.ones(N, T, 1, dtype=torch.float32, device="cuda") * 1.001
    A = torch.rand(N, T, 1, dtype=torch.float32, device="cuda") + 0.55
    X = torch.rand(N, T, D, dtype=torch.float32, device="cuda")

    #expand_compiled = torch.compile(expand_)

    @torch.inference_mode()
    def test_torch():        
        torch.cuda.synchronize()
        start = perf_counter_ns()
        for _ in range(100):    
            A1=A.clone()
            X1=X.clone()
            expand_(A1, X1)
        torch.cuda.synchronize()
        end = perf_counter_ns()
        print(f"torch: {(end-start)/1e6:.4}ms")
        return A1, X1

    @torch.inference_mode()
    def test_cuda():
        torch.cuda.synchronize()
        start = perf_counter_ns()
        for _ in range(100):
            A2 = A.clone()
            X2 = X.clone()
            z = pscan_scan_fwd(A2, X2)[0]
        torch.cuda.synchronize()
        end = perf_counter_ns()
        print(f"cuda: {(end-start)/1e6:.4}ms")
        return A2, X2

    A1,X1 = test_torch()
    A1,X1 = test_torch()
    A1,X1 = test_torch()
    A2,X2 = test_cuda()
    A2,X2 = test_cuda()
    A2,X2 = test_cuda()
    print("torch.allclose(A1, A2)", torch.allclose(A1, A2), torch.max(torch.abs(A1-A2)).item())
    print("torch.allclose(X1, X2)", torch.allclose(X1, X2), torch.max(torch.abs(X1-X2)).item())


def dev():
    #N, T, D = 1, 32, 4
    #N, T, D = 1, 8, 2*1024
    #N, T, D = 128, 256, 64  # works
    #N, T, D = 1024, 1024, 2 # works
    #N, T, D = 64*1024, 2048, 1 # works
    N, T, D = 256, 128, 256

    # A = torch.rand(N, T, 1, dtype=torch.float32, device="cuda") + 0.55
    # X = torch.rand(N, T, D, dtype=torch.float32, device="cuda")
    A = torch.ones(N, T, 1, dtype=torch.float32, device="cuda") * 0.4
    X = torch.ones(N, T, D, dtype=torch.float32, device="cuda")

    A1 = A.clone()
    X1 = X.clone()
    
    A2 = A.clone()
    X2 = X.clone()

    expand_(A1, X1)

    z = pscan_scan_fwd(A2, X2)[0]
    torch.cuda.synchronize()
    
    torch.set_printoptions(profile="full")
    #print('A', A[0,0:])
    #print('A_', A_)
    d = X1-X2
    d[d.abs()<0.0001] = 0
    
    #print('diff', d)
    #print(X2)
    print('A == A_', torch.allclose(A1, A2), torch.max(torch.abs(A1-A2)).item())
    print('X == X_', torch.allclose(X1, X2), torch.max(torch.abs(X1-X2)).item())

#dev()
benchmark()

