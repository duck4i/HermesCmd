#include <iostream>
#include <string>
#include "llama-cpp.h"

int main(int argc, char *argv[])
{
    std::string model_path = "NousResearch.Hermes-3-Llama-3.2-3B.Q4_K_M.gguf";
    std::cout << "HermesCLI - model " << model_path << " starting up." << std::endl;

    ggml_backend_load_all();

    std::cout << "GGML backend loaded" << std::endl;

    return 0;
}