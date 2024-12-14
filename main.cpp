#include <iostream>
#include <string>
#include "llama-cpp.h"

int main(int argc, char *argv[])
{
    std::string model_path = "NousResearch.Hermes-3-Llama-3.2-3B.Q4_K_M.gguf";

    std::cout << "HermesCLI - model " << model_path << " starting up." << std::endl;

    ggml_backend_load_all();

    std::cout << "GGML backend loaded" << std::endl;

    int n_predict = 32; // number of tokens to predict
    llama_model_params model_params = llama_model_default_params();

    llama_model *model = llama_load_model_from_file(model_path.c_str(), model_params);

    if (model == NULL)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    std::cout << "Model loaded OK." << std::endl;

    return 0;
}