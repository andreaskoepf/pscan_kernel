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


def expand(A, X):
    if A.size(1) == 1:
        return
    #print("expand_ ", A.shape, X.shape)
    T = 2 * (A.size(1) // 2)
    Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
    Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
    #print("Aa, Xa", Aa.shape, Xa.shape)
    Xaa = Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 0]))
    Aaa = Aa[:, :, 1].mul(Aa[:, :, 0])
    Aaa, Xaa = expand(Aaa, Xaa)

    Xa = Xa.clone()


    Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
    Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
    if T < A.size(1):
        #print('fixup:', T, A.size(1))
        X[:, -1].add_(A[:, -1].mul(X[:, -2]))
        A[:, -1].mul_(A[:, -2])


def benchmark():




    #N, T, D = 26, 16, 32
    #N, T, D = 128, 1024, 32
    #N, T, D = 256, 2048, 256
    N, T, D = 384, 1024, 64
    #N, T, D = 2, 1047, 3
    #N, T, D = 1024, 1024, 100

    #A = torch.ones(N, T, 1, dtype=torch.float32, device="cuda") * 1.001
    A = torch.rand(N, T, 1, dtype=torch.float32, device="cuda") + 0.55
    X = torch.rand(N, T, D, dtype=torch.float32, device="cuda")
    

    def capture_expand_graph(A, X):
        _compiled_inputs = tuple(v.clone() for v in (A, X))

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _ = expand_(*_compiled_inputs)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            expand_(*_compiled_inputs)
        
            def replay(A, X):
                _compiled_inputs[0].copy_(A)
                _compiled_inputs[1].copy_(X)
                g.replay()
                return _compiled_inputs[0].clone(), _compiled_inputs[1].clone()

        return replay
    
    # generate reference results
    A0,X0 = A.clone(), X.clone()
    expand_(A0, X0)

    #check graph version
    expand_replay = capture_expand_graph(A, X)
    # A5,X5 = expand_replay(A, X)
    # print("torch.allclose(A0, A5)", torch.allclose(A0, A5), torch.max(torch.abs(A0-A5)).item())
    # print("torch.allclose(X0, X5)", torch.allclose(X0, X5), torch.max(torch.abs(X0-X5)).item())

    @torch.inference_mode()
    @torch.no_grad()
    def benchmark(name, expand_fn):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):    
            A1=A.clone()
            X1=X.clone()
            expand_fn(A1, X1)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        print(f"{name}: {elapsed:.4}ms")
        return A1, X1

    for i in range(3):
        A1,X1 = benchmark("torch eager", expand_)
    # for i in range(3):
    #     _,_ = benchmark("graph replay", expand_replay)
    for i in range(3):
        A2,X2 = benchmark("custom cuda", pscan_scan_fwd)

    # expand_compiled = torch.compile(expand_, mode="reduce-overhead", fullgraph=True)
    # for i in range(3):
    #     A3,X3 = benchmark("torch compiled", expand_compiled)
    
    print("torch.allclose(A1, A2)", torch.allclose(A1, A2), torch.max(torch.abs(A1-A2)).item())
    print("torch.allclose(X1, X2)", torch.allclose(X1, X2), torch.max(torch.abs(X1-X2)).item())


def dev():
    #N, T, D = 1, 32, 4
    #N, T, D = 1, 8, 2*1024
    #N, T, D = 128, 256, 64  # works
    #N, T, D = 1024, 1024, 2 # works
    #N, T, D = 64*1024, 2048, 1 # works
    N, T, D = 256, 128, 256

    A = torch.randn(N, T, 1, dtype=torch.float32, device="cuda") * 0.1 + 1
    X = torch.rand(N, T, D, dtype=torch.float32, device="cuda")
    #A = torch.ones(N, T, 1, dtype=torch.float32, device="cuda") * 0.4
    #X = torch.ones(N, T, D, dtype=torch.float32, device="cuda")

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

