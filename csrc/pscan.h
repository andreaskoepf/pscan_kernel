#pragma once

struct PScanParams {
    int batch;
    int seqlen;
    int dim;
    
    torch::Tensor A;
    torch::Tensor X;
};
